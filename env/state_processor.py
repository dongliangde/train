"""
特征工程模块
将仿真原始数据转换为神经网络输入的标准化状态向量。

状态向量结构（每个时间步）:
  ┌─ 蓝方UAV特征 (每架UAV × UAV_FEATURES维) ─────────────────
  │  [距离, 方位角, 仰角, 速度, 航向, 高度, 威胁分, 在覆盖域内]
  ├─ 红方装备状态 (每个装备 × ASSET_FEATURES维) ──────────────
  │  [纬度归一, 经度归一, 最大射程归一, 是否就绪, 冷却剩余]
  └─ 全局特征 ──────────────────────────────────────────────
     [仿真时间归一, UAV数量归一, 最近威胁距离, 最高威胁分]
"""

import math
import numpy as np
from typing import Dict, List, Tuple

from .ws_client import RedAsset, BlueUAV


# ─────────────────────────────────────────────
# 地理计算工具
# ─────────────────────────────────────────────

EARTH_R = 6371000.0  # 地球半径（米）


def haversine(lat1, lon1, lat2, lon2) -> float:
    """返回两经纬度点间的球面距离（米）"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2) -> float:
    """从点1到点2的方位角（度，北为0，顺时针）"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def elevation_angle(dist_m: float, alt_diff_m: float) -> float:
    """从观测点到目标的仰角（度）"""
    if dist_m < 1e-3:
        return 90.0
    return math.degrees(math.atan2(alt_diff_m, dist_m))


def in_fov(az_to_target, el_to_target, asset_yaw, fov_az, fov_el) -> bool:
    """判断目标是否在装备视场内"""
    rel_az = (az_to_target - asset_yaw + 360) % 360
    if rel_az > 180:
        rel_az -= 360  # 转换到 [-180, 180]
    return (fov_az[0] <= rel_az <= fov_az[1]) and (fov_el[0] <= el_to_target <= fov_el[1])


# ─────────────────────────────────────────────
# 威胁评分
# ─────────────────────────────────────────────

# 保护目标（指挥所等重要节点），需根据实际场景配置
PROTECTED_TARGETS = [
    (39.910000, 116.460000),  # 示例：指挥所坐标
]

MAX_THREAT_DIST = 5000.0  # 威胁距离归一化基准（米）


def threat_score(uav: BlueUAV) -> float:
    """
    综合威胁评分 [0, 1]
    考虑：距保护目标距离 + 速度 + 高度（低空更危险）
    """
    min_dist = min(
        haversine(uav.lat, uav.lon, pt[0], pt[1])
        for pt in PROTECTED_TARGETS
    )
    dist_score  = max(0.0, 1.0 - min_dist / MAX_THREAT_DIST)
    speed_score = min(uav.speed / 30.0, 1.0)   # 30m/s 作为满分基准
    alt_score   = max(0.0, 1.0 - uav.alt / 200.0)  # 越低空威胁越高
    return 0.5 * dist_score + 0.3 * speed_score + 0.2 * alt_score


def predict_position(uav: BlueUAV, dt: float = 3.0) -> Tuple[float, float, float]:
    """简单线性外推预测dt秒后UAV位置（lat, lon, alt）"""
    if len(uav.history) < 2:
        return uav.lat, uav.lon, uav.alt
    # 用最近两帧估算速度矢量
    t1, la1, lo1, al1, _ = uav.history[-2]
    t2, la2, lo2, al2, _ = uav.history[-1]
    elapsed = max(t2 - t1, 1e-3)
    d_lat = (la2 - la1) / elapsed * dt
    d_lon = (lo2 - lo1) / elapsed * dt
    d_alt = (al2 - al1) / elapsed * dt
    return uav.lat + d_lat, uav.lon + d_lon, uav.alt + d_alt


# ─────────────────────────────────────────────
# 状态向量构建
# ─────────────────────────────────────────────

UAV_FEATURES   = 8   # 每架UAV的特征数
ASSET_FEATURES = 5   # 每个红方装备的特征数
GLOBAL_FEATURES = 4  # 全局特征数

MAX_UAVS   = 8   # 最大支持UAV数（不足时补零）
MAX_ASSETS = 10  # 最大支持红方装备数


class StateProcessor:
    """
    将 (red_assets, blue_uavs, sim_time) 转换为归一化numpy状态向量。
    obs_dim = MAX_UAVS*UAV_FEATURES + MAX_ASSETS*ASSET_FEATURES + GLOBAL_FEATURES
    """

    def __init__(self, ref_lat=39.91, ref_lon=116.46,
                 max_uavs=MAX_UAVS, max_assets=MAX_ASSETS):
        self.ref_lat   = ref_lat
        self.ref_lon   = ref_lon
        self.max_uavs  = max_uavs
        self.max_assets = max_assets
        self.obs_dim   = (max_uavs * UAV_FEATURES
                          + max_assets * ASSET_FEATURES
                          + GLOBAL_FEATURES)

    def compute(self,
                red_assets: Dict[str, RedAsset],
                blue_uavs:  Dict[str, BlueUAV],
                sim_time:   float) -> np.ndarray:

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0

        # ── 蓝方UAV特征 ───────────────────────────────
        uav_list = sorted(blue_uavs.values(),
                          key=lambda u: threat_score(u), reverse=True)

        for i in range(self.max_uavs):
            if i < len(uav_list):
                u = uav_list[i]
                # 用场景中心装备（第一个）作为参考观测点
                ref = next(iter(red_assets.values())) if red_assets else None
                if ref:
                    dist    = haversine(ref.lat, ref.lon, u.lat, u.lon)
                    az      = bearing(ref.lat, ref.lon, u.lat, u.lon)
                    el      = elevation_angle(dist, u.alt - ref.alt)
                    covered = int(in_fov(az, el, ref.yaw, ref.fov_az, ref.fov_el)
                                  and dist <= ref.max_range)
                else:
                    dist, az, el, covered = 5000.0, 0.0, 0.0, 0

                ts = threat_score(u)
                obs[idx:idx+UAV_FEATURES] = [
                    np.clip(dist / 5000.0, 0, 1),          # 距离归一
                    az / 360.0,                             # 方位角归一
                    (el + 90) / 180.0,                      # 仰角归一
                    np.clip(u.speed / 30.0, 0, 1),          # 速度归一
                    u.yaw / 360.0,                          # 无人机航向
                    np.clip(u.alt / 500.0, 0, 1),           # 高度归一
                    ts,                                     # 威胁评分
                    float(covered),                         # 是否在覆盖域
                ]
            idx += UAV_FEATURES

        # ── 红方装备状态 ──────────────────────────────
        asset_list = list(red_assets.values())[:self.max_assets]
        for i in range(self.max_assets):
            if i < len(asset_list):
                a = asset_list[i]
                obs[idx:idx+ASSET_FEATURES] = [
                    (a.lat - self.ref_lat + 1) / 2.0,    # 纬度归一
                    (a.lon - self.ref_lon + 1) / 2.0,    # 经度归一
                    np.clip(a.max_range / 10000.0, 0, 1), # 最大射程归一
                    float(a.is_ready),                    # 是否就绪
                    np.clip(a.cooldown_remaining / 30.0, 0, 1),  # 冷却归一
                ]
            idx += ASSET_FEATURES

        # ── 全局特征 ──────────────────────────────────
        n_uav = len(blue_uavs)
        scores = [threat_score(u) for u in blue_uavs.values()] if n_uav else [0.0]
        min_dist_to_protected = min(
            (haversine(u.lat, u.lon, pt[0], pt[1])
             for u in blue_uavs.values() for pt in PROTECTED_TARGETS),
            default=MAX_THREAT_DIST
        )
        obs[idx:idx+GLOBAL_FEATURES] = [
            np.clip(sim_time / 300.0, 0, 1),             # 仿真时间归一（5分钟）
            np.clip(n_uav / self.max_uavs, 0, 1),        # UAV数量比例
            np.clip(min_dist_to_protected / MAX_THREAT_DIST, 0, 1),  # 最近威胁距离
            max(scores),                                  # 最高威胁分
        ]

        return obs
