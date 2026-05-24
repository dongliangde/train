"""
WebSocket数据采集模块
对接仿真平台两个接口：
  - pushType=init     → 红方装备初始化（位置/能力参数）
  - pushType=realtime → 蓝方UAV实时态势推送
"""

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import websocket


# ─────────────────────────────────────────────
# 数据结构定义
# ─────────────────────────────────────────────

@dataclass
class RedAsset:
    """红方装备（雷达/光电/干扰设备等）"""
    name: str
    asset_type: str       # ty字段: 1212_radar_station / uav_navigation_site / ...
    lat: float
    lon: float
    alt: float
    yaw: float
    # 从force.params解析的关键能力参数
    max_range: float = 0.0         # 威力范围 / 最大作用距离（米）
    detect_range: float = 0.0      # 最大探测距离（km→m）
    charge_time: float = 0.0       # 充能时间（秒）
    fov_az: tuple = (-45.0, 45.0)  # 方位视场角 (下限, 上限)
    fov_el: tuple = (-5.0, 45.0)   # 俯仰视场角 (下限, 上限)
    # 状态
    is_ready: bool = True
    cooldown_remaining: float = 0.0


@dataclass
class BlueUAV:
    """蓝方无人机实时状态"""
    name: str
    lat: float
    lon: float
    alt: float
    yaw: float
    pitch: float
    roll: float
    speed: float
    timestamp: float = 0.0
    # 轨迹历史（用于运动预测）
    history: deque = field(default_factory=lambda: deque(maxlen=20))

    def update(self, data: dict, ts: float):
        self.lat   = data["la"]
        self.lon   = data["lo"]
        self.alt   = data["al"]
        self.yaw   = data["ya"]
        self.pitch = data["pi"]
        self.roll  = data["ro"]
        self.speed = data["sp"]
        self.timestamp = ts
        self.history.append((ts, self.lat, self.lon, self.alt, self.speed))


# ─────────────────────────────────────────────
# 参数解析工具函数
# ─────────────────────────────────────────────

def _extract_param(params: list, name: str, default=0.0):
    """从force.params列表中按name提取value"""
    for p in params:
        if p.get("name") == name:
            try:
                return float(p["value"])
            except (TypeError, ValueError):
                return default
    return default


def _extract_fov(params: list, name: str) -> tuple:
    """提取视场角上下限"""
    for p in params:
        if p.get("name") == name and p.get("children"):
            lo = hi = 0.0
            for c in p["children"]:
                if c["name"] == "下限":
                    lo = float(c["value"] or 0)
                elif c["name"] == "上限":
                    hi = float(c["value"] or 0)
            return (lo, hi)
    return (-45.0, 45.0)


def parse_red_asset(item: dict) -> Optional[RedAsset]:
    """解析init接口中一个红方装备对象"""
    if item.get("si") != "red":
        return None
    force  = item.get("force", {})
    params = force.get("params", [])
    # 组件参数（传感器）
    components = force.get("components", [])
    sensor_params = []
    if components:
        sensor_params = components[0].get("params", [])

    asset = RedAsset(
        name       = item["na"],
        asset_type = item["ty"],
        lat        = item["la"],
        lon        = item["lo"],
        alt        = item["al"],
        yaw        = item["ya"],
    )

    # 按设备类型提取关键参数
    ty = item["ty"]
    if "radar" in ty:
        asset.max_range    = _extract_param(params, "威力范围")
        asset.detect_range = _extract_param(sensor_params, "最大探测距离") * 1000  # km→m
        asset.fov_az       = _extract_fov(sensor_params, "方位视场角")
        asset.fov_el       = _extract_fov(sensor_params, "俯仰视场角")
    elif "navigation" in ty or "uav_navigation" in ty:
        asset.max_range   = _extract_param(params, "最大作用距离")
        asset.charge_time = _extract_param(params, "充能时间")
        asset.fov_az      = _extract_fov(sensor_params, "方位视场角")
        asset.fov_el      = _extract_fov(sensor_params, "俯仰视场角")
    else:
        asset.max_range = _extract_param(params, "威力范围") or _extract_param(params, "最大作用距离")

    return asset


# ─────────────────────────────────────────────
# WebSocket 客户端
# ─────────────────────────────────────────────

class SimulationClient:
    """
    管理与仿真平台的两条WebSocket连接，
    维护红方装备字典和蓝方UAV状态字典。
    """

    INIT_URL     = "ws://127.0.0.1:38838/api/v1/data-socket/getModelData?pushType=init"
    REALTIME_URL = "ws://127.0.0.1:38838/api/v1/data-socket/getModelData?pushType=realtime"

    def __init__(self):
        self.red_assets: Dict[str, RedAsset] = {}   # name → RedAsset
        self.blue_uavs:  Dict[str, BlueUAV]  = {}   # name → BlueUAV
        self.sim_time: float = 0.0
        self.initialized: bool = False
        self._lock = threading.Lock()
        self._ws_init = None
        self._ws_rt   = None

    # ── init接口 ──────────────────────────────

    def _on_init_message(self, ws, raw):
        data = json.loads(raw)
        assets = {}
        for item in data.get("init_model", []):
            asset = parse_red_asset(item)
            if asset:
                assets[asset.name] = asset
            elif item.get("si") == "blue":
                # 蓝方init信息（位置可做参考起点）
                name = item["na"]
                with self._lock:
                    self.blue_uavs[name] = BlueUAV(
                        name  = name,
                        lat   = item["la"],
                        lon   = item["lo"],
                        alt   = item["al"],
                        yaw   = item["ya"],
                        pitch = item["pi"],
                        roll  = item["ro"],
                        speed = item["sp"],
                    )
        with self._lock:
            self.red_assets = assets
            self.initialized = True
        print(f"[Init] 红方装备 {len(self.red_assets)} 个: {list(self.red_assets.keys())}")

    # ── realtime接口 ──────────────────────────

    def _on_realtime_message(self, ws, raw):
        data = json.loads(raw)
        ts   = data.get("ts", 0.0)
        with self._lock:
            self.sim_time = ts
            for item in data.get("up", []):
                name = item["na"]
                if name not in self.blue_uavs:
                    self.blue_uavs[name] = BlueUAV(
                        name=name, lat=item["la"], lon=item["lo"],
                        alt=item["al"], yaw=item["ya"],
                        pitch=item["pi"], roll=item["ro"], speed=item["sp"]
                    )
                self.blue_uavs[name].update(item, ts)

    # ── 连接管理 ──────────────────────────────

    def _make_ws(self, url, on_message):
        return websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=lambda ws, e: print(f"[WS Error] {url}: {e}"),
            on_close=lambda ws, c, m: print(f"[WS Closed] {url}"),
        )

    def connect(self):
        """启动两条WS连接（各自独立线程）"""
        self._ws_init = self._make_ws(self.INIT_URL, self._on_init_message)
        self._ws_rt   = self._make_ws(self.REALTIME_URL, self._on_realtime_message)

        threading.Thread(target=self._ws_init.run_forever, daemon=True).start()
        threading.Thread(target=self._ws_rt.run_forever,   daemon=True).start()

        # 等待init完成
        timeout = 10.0
        start   = time.time()
        while not self.initialized and (time.time() - start) < timeout:
            time.sleep(0.1)
        if not self.initialized:
            raise TimeoutError("仿真平台init接口超时，请检查连接")

    def get_snapshot(self):
        """线程安全地获取当前全量状态快照"""
        with self._lock:
            return (
                dict(self.red_assets),
                dict(self.blue_uavs),
                self.sim_time,
            )
