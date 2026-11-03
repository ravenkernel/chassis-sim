"""传感器层演示：IMU / 轮速计 / 参数化 LiDAR + 延迟与丢帧。

运行：/home/zhao/anaconda3/envs/drl2/bin/python examples/sensors_demo.py

传感器只写进 info、不进 obs：本脚本直接读 bridge 的 get_sensor_reading()，
env 路径等价于 info["imu"] / info["wheel"] / info["scan"] / info["scan_meta"]。
"""

from __future__ import annotations

import numpy as np

from chassis_sim.bridges import SimpleBridge
from chassis_sim.bridges.sensors import LidarParams, SensorConfig
from chassis_sim.chassis import AckermannChassis

_PATH = np.column_stack([np.linspace(1.0, 6.0, 40), np.full(40, 2.0), np.zeros(40)])
_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.30}, "state": [3.0, 2.5, 0.0]},
]


def make_bridge(**lidar_kw) -> SimpleBridge:
    cfg = SensorConfig(
        lidar=LidarParams(**lidar_kw),
        imu=True,
        wheel=True,
        imu_noise_std=0.02,
        wheel_noise_std=0.01,
        sensor_seed=2026,
    )
    bridge = SimpleBridge(AckermannChassis(), sensors=cfg)
    bridge.reset(
        obstacles=_OBSTACLES,
        path_waypoints=_PATH.copy(),
        initial_heading=0.0,
    )
    return bridge


def fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"


def main() -> None:
    bridge = make_bridge(noise_std=0.03)
    print("=== 20 步：IMU / 轮速计 / 参数化 LiDAR（180 线, FOV=pi, 10 m）===")
    for k in range(1, 21):
        bridge.step(np.array([0.05, 0.5], dtype=np.float64))
        reading = bridge.get_sensor_reading()
        imu = reading["imu"]
        wheel = reading["wheel"]
        scan = reading["scan"]
        meta = reading["scan_meta"]
        print(
            f"step {k:2d} t={imu['t']:.1f} | imu ax={imu['ax']:+.6f} "
            f"ay={imu['ay']:+.6f} yaw_rate={imu['yaw_rate']:+.6f} yaw={imu['yaw']:+.6f}"
        )
        print(
            f"          wheel v_left={wheel['v_left']:+.6f} "
            f"v_right={wheel['v_right']:+.6f} v={wheel['v']:+.6f}"
        )
        print(
            f"          scan{scan.shape} dtype={scan.dtype} 前3={fmt(scan[:3])} "
            f"angle_increment={meta['angle_increment']:.12f}"
        )

    print()
    print("=== 延迟 latency_steps=3：第 k 步交付第 k-3 步采集的帧 ===")
    delayed = make_bridge(latency_steps=3)
    for k in range(1, 8):
        delayed.step(np.array([0.0, 0.5], dtype=np.float64))
        meta = delayed.get_sensor_reading()["scan_meta"]
        print(
            f"step {k} t={meta['t']:.1f} valid={meta['valid']} "
            f"frame_t={meta['frame_t']} dropped={meta['dropped']}"
        )

    print()
    print("=== 丢帧 dropout_prob=1.0：scan 恒为 None，其余读数照常 ===")
    dropping = make_bridge(dropout_prob=1.0)
    for k in range(1, 4):
        dropping.step(np.array([0.0, 0.5], dtype=np.float64))
        reading = dropping.get_sensor_reading()
        print(
            f"step {k} scan={reading['scan']} "
            f"dropped={reading['scan_meta']['dropped']} v={reading['wheel']['v']:+.6f}"
        )


if __name__ == "__main__":
    main()
