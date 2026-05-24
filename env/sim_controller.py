"""
运控接口封装
对接 HTTP 控制接口，管理仿真回合的生命周期。

接口地址:
  POST /api/v1/data-server/init     → 初始化想定
  POST /api/v1/data-server/control  → 运控指令
"""

import time
import requests
from dataclasses import dataclass
from typing import Optional


BASE_URL = "http://127.0.0.1:38838"


# ─── cmd 枚举 ────────────────────────────────────────────────────────────────

class CMD:
    START          = 0
    SUSPEND        = 1
    CONTINUE       = 2
    STOP           = 3
    CHARACTERISATION = 4   # 改变倍速
    SCHEDULE       = 5     # 跳转进度（仅回放）
    RESET          = 6
    CHANGESET      = 7     # 改变步长


# ─── 控制器 ──────────────────────────────────────────────────────────────────

class SimulationController:
    """
    封装仿真平台的 HTTP 运控接口。
    负责每一局训练的 初始化→启动→加速→重置 完整流程。
    """

    def __init__(self,
                 scenario_id: int,
                 speed_ratio: float = 10.0,
                 sim_step: float = 0.1,
                 base_url: str = BASE_URL,
                 timeout: float = 5.0):
        """
        Args:
            scenario_id: 想定ID（对应仿真平台中的场景编号）
            speed_ratio: 仿真加速比，10.0 表示10倍速（训练时尽量调高）
            sim_step:    仿真步长（秒），影响推送频率
            base_url:    服务地址
            timeout:     HTTP 请求超时（秒）
        """
        self.scenario_id = scenario_id
        self.speed_ratio = speed_ratio
        self.sim_step    = sim_step
        self.base_url    = base_url
        self.timeout     = timeout
        self._session    = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ─── 底层 HTTP ───────────────────────────────────────────────────────────

    def _post_init(self) -> bool:
        url  = f"{self.base_url}/api/v1/data-server/init"
        body = {
            "id":   self.scenario_id,
            "mode": 0,          # 仿真模式
            "config": {
                "mode":         False,   # 事件模式
                "automatic":    True,
                "interval":     300,
                "unit":         0,
                "replayRecord": True,
            }
        }
        try:
            r = self._session.post(url, json=body, timeout=self.timeout)
            data = r.json()
            ok = data.get("code") == 200
            if not ok:
                print(f"[Controller] Init 失败: {data.get('msg')}")
            return ok
        except Exception as e:
            print(f"[Controller] Init 请求异常: {e}")
            return False

    def _post_control(self, cmd: int, **kwargs) -> bool:
        url  = f"{self.base_url}/api/v1/data-server/control"
        body = {"cmd": cmd, **kwargs}
        try:
            r = self._session.post(url, json=body, timeout=self.timeout)
            data = r.json()
            ok = data.get("code") == 200
            if not ok:
                print(f"[Controller] CMD={cmd} 失败: {data.get('msg')}")
            return ok
        except Exception as e:
            print(f"[Controller] CMD={cmd} 请求异常: {e}")
            return False

    # ─── 高层流程 ────────────────────────────────────────────────────────────

    def start_episode(self) -> bool:
        """
        开始新一局：RESET → INIT → SET_STEP → SET_SPEED → START
        返回是否成功。
        """
        # 1. 重置（清空上一局状态）
        self._post_control(CMD.RESET)
        time.sleep(0.3)  # 等待平台重置完成

        # 2. 重新初始化想定
        if not self._post_init():
            return False
        time.sleep(0.3)

        # 3. 设置仿真步长
        self._post_control(CMD.CHANGESET, step=self.sim_step)

        # 4. 设置加速比（训练时可大幅提速）
        if self.speed_ratio != 1.0:
            self._post_control(CMD.CHARACTERISATION, speedRatio=self.speed_ratio)

        # 5. 启动
        ok = self._post_control(CMD.START)
        if ok:
            print(f"[Controller] 新一局开始 "
                  f"(场景={self.scenario_id}, 倍速={self.speed_ratio}x, "
                  f"步长={self.sim_step}s)")
        return ok

    def stop_episode(self):
        """结束当前局"""
        self._post_control(CMD.STOP)

    def pause(self):
        self._post_control(CMD.SUSPEND)

    def resume(self):
        self._post_control(CMD.CONTINUE)

    def set_speed(self, ratio: float):
        """动态调整仿真速度"""
        self.speed_ratio = ratio
        self._post_control(CMD.CHARACTERISATION, speedRatio=ratio)
        print(f"[Controller] 倍速 → {ratio}x")
