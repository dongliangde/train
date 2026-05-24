# 红方智能体训练框架

## 项目结构

```
red_agent/
├── env/
│   ├── ws_client.py        # WebSocket 数据采集（init + realtime）
│   ├── state_processor.py  # 特征工程 → 归一化状态向量
│   └── simulation_env.py   # Gymnasium 兼容的 RL 环境
├── training/
│   └── train.py            # PPO 训练 / 数据采集 / 推理入口
└── README.md
```

## 安装依赖

```bash
pip install gymnasium stable-baselines3 websocket-client numpy
# 可选：TensorBoard 日志可视化
pip install tensorboard
```

## 快速开始

### 1. 启动仿真平台
确保仿真平台 WebSocket 服务运行在 `127.0.0.1:38838`

### 2. 训练
```bash
python -m red_agent.training.train --mode train
```

### 3. 仅采集数据（随机策略录制轨迹）
```bash
python -m red_agent.training.train --mode collect --episodes 50
```

### 4. 加载模型推理
```bash
python -m red_agent.training.train --mode infer --model ./models/red_agent/red_agent_final
```

### 5. TensorBoard 监控
```bash
tensorboard --logdir ./logs/red_agent
```

## 关键配置说明

### 保护目标坐标
在 `env/state_processor.py` 中修改 `PROTECTED_TARGETS`：
```python
PROTECTED_TARGETS = [
    (39.910000, 116.460000),  # 指挥所坐标（经纬度）
    (39.915000, 116.465000),  # 其他重要目标
]
```

### 奖励函数调优
在 `env/simulation_env.py` 中调整奖励常数：
```python
R_ENGAGE_SUCCESS = 10.0   # 成功压制UAV的奖励
R_UAV_IN_ZONE   = -20.0  # UAV进入保护区的惩罚
R_UAV_DESTROYED =  15.0  # UAV被消灭的奖励
```

### 接入仿真平台控制接口
在 `env/simulation_env.py` 的 `_execute_actions` 方法中，
将注释处替换为实际的控制指令 HTTP/WebSocket 发送逻辑：
```python
# 示例：发送打击指令到仿真平台
requests.post(f"http://127.0.0.1:38838/api/v1/engage",
              json={"asset": asset.name, "target": target.name})
```

## 状态向量结构（obs_dim = 124维）

| 分组 | 维度 | 说明 |
|------|------|------|
| 蓝方UAV × 8 | 64维 | 距离/方位/仰角/速度/航向/高度/威胁分/覆盖域 |
| 红方装备 × 10 | 50维 | 位置/射程/就绪状态/冷却时间 |
| 全局特征 | 4维 | 仿真时间/UAV数量/最近威胁距离/最高威胁分 |

## 扩展方向

- **多智能体**：每个红方装备独立 Agent，使用 MAPPO
- **图神经网络**：将装备-UAV关系建模为图，替换 MLP
- **课程学习**：从单架UAV开始，逐步增加数量和战术复杂度
- **行为克隆**：先用专家数据预训练，再用 RL 微调
