"""
训练入口
支持三种模式：
  train   → PPO 多局连续训练
  collect → 随机策略录制轨迹数据
  infer   → 加载已训练模型推理评估

安装依赖：
  pip install stable-baselines3 gymnasium websocket-client requests
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# 将项目根目录加入 sys.path，支持直接运行此文件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from env.simulation_env import RedVsBlueEnv


# ─── 训练配置 ─────────────────────────────────────────────────────────────────

CFG = {
    # 想定参数（从系统状态接口获取 simulateId）
    "scenario_id":   95,
    "speed_ratio":   10.0,    # 仿真加速倍数（训练时建议 10~20）
    "sim_step":      0.1,     # 仿真步长（秒）

    # 环境参数
    "max_steps":     500,     # 每局最大决策步数
    "max_assets":    10,
    "max_uavs":      8,

    # PPO 超参数
    "learning_rate": 3e-4,
    "n_steps":       256,     # 每次 rollout 收集步数（多局累积）
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.01,    # 熵正则（鼓励探索）
    "vf_coef":       0.5,

    # 训练规模
    "total_timesteps": 200_000,   # 总训练步数

    # 输出
    "log_dir":   "./logs/red_agent",
    "model_dir": "./models/red_agent",
    "save_freq": 5_000,
}


# ─── 训练回调 ─────────────────────────────────────────────────────────────────

class EpisodeLogger(BaseCallback):
    """每局结束时打印统计，定期记录到 TensorBoard"""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.ep_rewards  = []
        self.ep_lengths  = []
        self.ep_n_uav    = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep  = info["episode"]
            r   = ep["r"]
            l   = ep["l"]
            nuav = info.get("n_uav", 0)

            self.ep_rewards.append(r)
            self.ep_lengths.append(l)
            self.ep_n_uav.append(nuav)

            # TensorBoard 自定义指标
            self.logger.record("rollout/ep_rew_mean_20",
                                np.mean(self.ep_rewards[-20:]))
            self.logger.record("rollout/ep_len_mean_20",
                                np.mean(self.ep_lengths[-20:]))
            self.logger.record("custom/episodes_total",
                                len(self.ep_rewards))

            if self.verbose and len(self.ep_rewards) % 5 == 0:
                print(
                    f"[Ep {len(self.ep_rewards):4d} | "
                    f"Step {self.num_timesteps:7d}] "
                    f"Rew={r:8.2f}  "
                    f"Len={l:4d}  "
                    f"UAV残余={nuav}  "
                    f"近20均值={np.mean(self.ep_rewards[-20:]):8.2f}"
                )
        return True


# ─── 环境工厂 ─────────────────────────────────────────────────────────────────

def make_env():
    env = RedVsBlueEnv(
        scenario_id  = CFG["scenario_id"],
        speed_ratio  = CFG["speed_ratio"],
        sim_step     = CFG["sim_step"],
        max_steps    = CFG["max_steps"],
        max_assets   = CFG["max_assets"],
        max_uavs     = CFG["max_uavs"],
    )
    return Monitor(env)


# ─── 训练主流程 ───────────────────────────────────────────────────────────────

def train():
    if not SB3_AVAILABLE:
        print("请先安装: pip install stable-baselines3")
        return

    os.makedirs(CFG["log_dir"],   exist_ok=True)
    os.makedirs(CFG["model_dir"], exist_ok=True)

    print("=" * 65)
    print(f"  红方智能体 PPO 训练")
    print(f"  场景ID={CFG['scenario_id']}  加速={CFG['speed_ratio']}x  "
          f"总步数={CFG['total_timesteps']:,}")
    print("=" * 65)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = CFG["learning_rate"],
        n_steps         = CFG["n_steps"],
        batch_size      = CFG["batch_size"],
        n_epochs        = CFG["n_epochs"],
        gamma           = CFG["gamma"],
        gae_lambda      = CFG["gae_lambda"],
        clip_range      = CFG["clip_range"],
        ent_coef        = CFG["ent_coef"],
        vf_coef         = CFG["vf_coef"],
        verbose         = 0,
        tensorboard_log = CFG["log_dir"],
        policy_kwargs   = dict(net_arch=[256, 256, 128]),
    )

    print(f"  观测维度: {env.observation_space.shape[0]}")
    print(f"  动作空间: {env.action_space.nvec.tolist()}")
    print("-" * 65)

    model.learn(
        total_timesteps = CFG["total_timesteps"],
        callback        = [
            EpisodeLogger(verbose=1),
            CheckpointCallback(
                save_freq   = CFG["save_freq"],
                save_path   = CFG["model_dir"],
                name_prefix = "red_agent",
                verbose     = 0,
            ),
        ],
        progress_bar    = True,
        reset_num_timesteps = True,
    )

    # 保存最终模型
    out = os.path.join(CFG["model_dir"], "red_agent_final")
    model.save(out)
    env.save(out + "_vecnorm.pkl")
    print(f"\n[Done] 模型保存至 {out}")
    print(f"  TensorBoard: tensorboard --logdir {CFG['log_dir']}")


# ─── 数据采集模式 ─────────────────────────────────────────────────────────────

def collect_data(n_episodes: int = 30):
    """
    随机策略跑多局，录制 (obs, action, reward, next_obs) 轨迹。
    用于：行为克隆预训练 / 离线RL数据集构建。
    """
    save_dir = Path("./data/trajectories")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()
    print(f"[Collect] 开始采集 {n_episodes} 局数据...")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_data = []

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_data.append({
                "obs":      obs.tolist(),
                "action":   action.tolist(),
                "reward":   float(reward),
                "next_obs": next_obs.tolist(),
                "done":     terminated or truncated,
                "sim_time": info.get("sim_time", 0),
                "n_uav":    info.get("n_uav", 0),
            })
            obs  = next_obs
            done = terminated or truncated

        total_r = sum(d["reward"] for d in ep_data)
        fname   = save_dir / f"ep_{ep:04d}.npy"
        np.save(fname, ep_data)
        print(f"  Episode {ep+1:3d}/{n_episodes} | "
              f"Steps={len(ep_data):4d} | "
              f"TotalReward={total_r:8.2f} | → {fname.name}")

    env.close()
    print(f"[Collect] 完成，数据保存至 {save_dir}")


# ─── 推理评估 ─────────────────────────────────────────────────────────────────

def run_inference(model_path: str, n_episodes: int = 5):
    if not SB3_AVAILABLE:
        return

    env   = DummyVecEnv([make_env])
    env   = VecNormalize.load(model_path + "_vecnorm.pkl", env)
    env.training    = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    print(f"[Infer] 加载模型: {model_path}")
    print(f"[Infer] 运行 {n_episodes} 局评估（倍速={CFG['speed_ratio']}x）")

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        total_r, steps = 0.0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_r += float(reward[0])
            steps   += 1

        print(f"  Episode {ep+1}: Steps={steps:4d} | TotalReward={total_r:8.2f}")

    env.close()


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="红方智能体")
    parser.add_argument("--mode", choices=["train", "collect", "infer"],
                        default="train")
    parser.add_argument("--model",
                        default="./models/red_agent/red_agent_final",
                        help="infer 模式下的模型路径")
    parser.add_argument("--episodes", type=int, default=30,
                        help="collect 模式的采集局数")
    parser.add_argument("--speed", type=float, default=None,
                        help="覆盖 CFG 中的仿真加速比")
    args = parser.parse_args()

    if args.speed:
        CFG["speed_ratio"] = args.speed

    if args.mode == "train":
        train()
    elif args.mode == "collect":
        collect_data(n_episodes=args.episodes)
    elif args.mode == "infer":
        run_inference(args.model, n_episodes=args.episodes)
