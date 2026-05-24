"""
红方强化学习环境（Gymnasium 兼容）

回合流程：
  reset()  →  Controller: RESET → INIT → START
  step()   →  等待仿真时间推进 → 执行动作 → 计算奖励
  done     →  Controller: STOP  → 返回终止信号

动作空间（MultiDiscrete）：
  对每个红方装备独立决策
  0 = STANDBY   待机
  1 = ENGAGE    打击/干扰威胁最高的UAV
  2 = ENGAGE_2  打击/干扰威胁次高的UAV
  3 = TOGGLE    切换工作状态

奖励函数：
  + 成功压制/摧毁无人机（按威胁等级加权）
  - 无人机进入保护区域
  - 超范围/冷却中的无效打击
  - 每步微小惩罚（鼓励快速决策）
"""

import time
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple

from .ws_client import SimulationClient, RedAsset, BlueUAV
from .state_processor import StateProcessor, haversine, threat_score, PROTECTED_TARGETS
from .sim_controller import SimulationController


# ─── 动作常量 ─────────────────────────────────────────────────────────────────

ACTION_STANDBY  = 0
ACTION_ENGAGE   = 1
ACTION_ENGAGE_2 = 2
ACTION_TOGGLE   = 3
N_ACTIONS       = 4


# ─── 奖励常数 ─────────────────────────────────────────────────────────────────

R_ENGAGE_SUCCESS  =  10.0   # 有效打击（按威胁分加权）
R_ENGAGE_FAIL     =  -1.0   # 无效打击（超范围 / 冷却中）
R_UAV_IN_ZONE     = -20.0   # UAV进入保护区
R_UAV_DESTROYED   =  15.0   # UAV从仿真中消失（视为摧毁）
R_HIGH_THREAT     =  -0.3   # 每步高威胁UAV存活惩罚
R_STEP            =  -0.05  # 每步固定惩罚
PROTECTED_RADIUS  = 500.0   # 保护区半径（米）


# ─── 环境主体 ─────────────────────────────────────────────────────────────────

class RedVsBlueEnv(gym.Env):
    """
    红方对抗蓝方低空无人机的强化学习环境。

    使用方式：
        env = RedVsBlueEnv(scenario_id=95, speed_ratio=10.0)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 scenario_id: int = 95,
                 speed_ratio: float = 10.0,
                 sim_step: float = 0.1,
                 max_steps: int = 500,
                 step_wait_timeout: float = 5.0,
                 max_assets: int = 10,
                 max_uavs: int = 8):
        """
        Args:
            scenario_id:        想定ID（从系统状态接口 simulateId 字段获取）
            speed_ratio:        仿真加速比（训练时建议10~20）
            sim_step:           仿真步长（秒）
            max_steps:          每局最大步数
            step_wait_timeout:  等待仿真时间推进的超时（秒，实际时间，非仿真时间）
            max_assets:         最大红方装备数
            max_uavs:           最大蓝方UAV数
        """
        super().__init__()

        self.max_steps          = max_steps
        self.step_wait_timeout  = step_wait_timeout

        # 子模块
        self.client     = SimulationClient()
        self.controller = SimulationController(
            scenario_id = scenario_id,
            speed_ratio = speed_ratio,
            sim_step    = sim_step,
        )
        self.processor = StateProcessor(max_uavs=max_uavs, max_assets=max_assets)

        # 回合内状态
        self._step_count    = 0
        self._prev_uav_set  = set()
        self._last_sim_time = -1.0
        self._episode_count = 0

        # ── 观测空间 ─────────────────────────────────────────────────────────
        obs_dim = self.processor.obs_dim
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ── 动作空间：每个装备槽位独立决策 ──────────────────────────────────
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * max_assets)

    # ─── Gym 接口 ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # 首次连接 WebSocket
        if not self.client.initialized:
            self.client.connect()

        # 结束上一局（若存在）
        if self._episode_count > 0:
            self.controller.stop_episode()
            time.sleep(0.2)

        # 启动新一局
        ok = self.controller.start_episode()
        if not ok:
            raise RuntimeError("仿真平台启动失败，请检查运控接口")

        # 等待 WebSocket 推送第一帧数据
        self._wait_for_first_frame()

        # 初始化回合状态
        self._step_count   = 0
        self._prev_uav_set = set(self.client.blue_uavs.keys())
        self._last_sim_time = self.client.sim_time
        self._episode_count += 1

        obs = self._get_obs()
        info = {"episode": self._episode_count, "scenario_id": self.controller.scenario_id}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 1. 等待仿真时间推进（避免在同一帧重复决策）
        self._wait_for_sim_advance()

        red_assets, blue_uavs, sim_time = self.client.get_snapshot()

        # 2. 执行动作
        exec_reward = self._execute_actions(action, red_assets, blue_uavs)

        # 3. 计算环境奖励
        env_reward = self._compute_env_reward(blue_uavs)

        # 4. 构建新观测
        obs = self.processor.compute(red_assets, blue_uavs, sim_time)

        reward = exec_reward + env_reward + R_STEP
        self._step_count   += 1
        self._last_sim_time = sim_time

        terminated = self._check_terminated(blue_uavs)
        truncated  = self._step_count >= self.max_steps

        if terminated or truncated:
            self.controller.stop_episode()

        info = {
            "step":      self._step_count,
            "sim_time":  sim_time,
            "n_uav":     len(blue_uavs),
            "episode":   self._episode_count,
            "reward_breakdown": {
                "exec": round(exec_reward, 3),
                "env":  round(env_reward, 3),
            }
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.controller.stop_episode()

    # ─── 等待机制 ─────────────────────────────────────────────────────────────

    def _wait_for_first_frame(self, timeout: float = 8.0):
        """等待 realtime WS 推送第一帧数据"""
        start = time.time()
        while self.client.sim_time <= 0 and (time.time() - start) < timeout:
            time.sleep(0.05)
        if self.client.sim_time <= 0:
            print("[Warning] 未收到仿真数据，检查平台是否正常运行")

    def _wait_for_sim_advance(self):
        """
        等待仿真时间推进至少一个步长。
        由于加速比可能很高，实际等待时间很短。
        """
        start    = time.time()
        baseline = self._last_sim_time
        while True:
            _, _, current_ts = self.client.get_snapshot()
            if current_ts > baseline + 1e-6:
                return
            if (time.time() - start) > self.step_wait_timeout:
                print(f"[Warning] 等待仿真推进超时 (sim_time={current_ts:.3f})")
                return
            time.sleep(0.01)  # 轮询间隔10ms（实际时间，与仿真加速无关）

    # ─── 动作执行 ─────────────────────────────────────────────────────────────

    def _execute_actions(self,
                         action: np.ndarray,
                         red_assets: Dict[str, RedAsset],
                         blue_uavs: Dict[str, BlueUAV]) -> float:
        """
        解析动作数组，模拟执行并返回即时奖励。
        ⚡ 若仿真平台提供打击/干扰的 HTTP 指令接口，在此处替换为实际调用。
        """
        total_reward = 0.0
        asset_list   = list(red_assets.values())
        uav_sorted   = sorted(blue_uavs.values(), key=threat_score, reverse=True)

        for i, asset in enumerate(asset_list):
            if i >= len(action):
                break
            act = int(action[i])

            if act == ACTION_STANDBY:
                continue

            elif act in (ACTION_ENGAGE, ACTION_ENGAGE_2):
                uav_idx = 0 if act == ACTION_ENGAGE else 1
                if uav_idx >= len(uav_sorted):
                    total_reward += R_ENGAGE_FAIL
                    continue

                target = uav_sorted[uav_idx]
                dist   = haversine(asset.lat, asset.lon, target.lat, target.lon)

                if not asset.is_ready:
                    total_reward += R_ENGAGE_FAIL   # 冷却中无效打击
                elif dist > asset.max_range:
                    total_reward += R_ENGAGE_FAIL   # 超出范围
                else:
                    ts = threat_score(target)
                    total_reward += R_ENGAGE_SUCCESS * (0.5 + 0.5 * ts)
                    # 触发冷却
                    asset.is_ready = False
                    asset.cooldown_remaining = asset.charge_time or 5.0

            elif act == ACTION_TOGGLE:
                asset.is_ready = not asset.is_ready

        # 更新冷却（基于仿真时间差，非实际时间）
        dt = self.client.sim_time - self._last_sim_time
        if dt > 0:
            for asset in red_assets.values():
                if not asset.is_ready:
                    asset.cooldown_remaining -= dt
                    if asset.cooldown_remaining <= 0:
                        asset.is_ready = True
                        asset.cooldown_remaining = 0.0

        return total_reward

    # ─── 奖励计算 ─────────────────────────────────────────────────────────────

    def _compute_env_reward(self, blue_uavs: Dict[str, BlueUAV]) -> float:
        reward = 0.0
        current_set = set(blue_uavs.keys())

        # UAV消失视为被摧毁
        destroyed = self._prev_uav_set - current_set
        reward += len(destroyed) * R_UAV_DESTROYED

        # UAV进入保护区惩罚
        for uav in blue_uavs.values():
            for pt in PROTECTED_TARGETS:
                if haversine(uav.lat, uav.lon, pt[0], pt[1]) < PROTECTED_RADIUS:
                    reward += R_UAV_IN_ZONE * (0.5 + 0.5 * threat_score(uav))
                    break

        # 高威胁UAV持续存活惩罚（按仿真时间差加权）
        dt = max(self.client.sim_time - self._last_sim_time, 0)
        high_threat = [u for u in blue_uavs.values() if threat_score(u) > 0.7]
        reward += len(high_threat) * R_HIGH_THREAT * min(dt, 5.0)

        self._prev_uav_set = current_set
        return reward

    # ─── 终止判断 ─────────────────────────────────────────────────────────────

    def _check_terminated(self, blue_uavs: Dict[str, BlueUAV]) -> bool:
        if len(blue_uavs) == 0:
            print(f"[Env] Episode {self._episode_count} 终止：全部UAV已清除 ✓")
            return True
        for uav in blue_uavs.values():
            for pt in PROTECTED_TARGETS:
                if haversine(uav.lat, uav.lon, pt[0], pt[1]) < PROTECTED_RADIUS * 0.3:
                    print(f"[Env] Episode {self._episode_count} 终止：UAV {uav.name} 突破防线 ✗")
                    return True
        return False

    def _get_obs(self) -> np.ndarray:
        red, blue, ts = self.client.get_snapshot()
        return self.processor.compute(red, blue, ts)
