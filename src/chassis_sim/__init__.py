"""chassis-sim：三种底盘的 Gymnasium 环境。

import 本包即注册三个 id（ChassisSim-Ackermann-v1 / ChassisSim-DiffDrive-v1 /
ChassisSim-Mecanum-v1）。注册逻辑在 gym_registration.py，只依赖 gymnasium：
环境本体按 entry_point 字符串延迟 import，不拉起 numba / matplotlib / torch。

1.0.0 起顶层就是公共 API：配置（EnvConfig / RewardConfig）、统一入口 make()、
环境本体、观测 spec、传感器、控制器、评估与场景一览都在这里 re-export；
实现仍住在各自子包（api.py / config/schema.py / envs/ ...），本文件不写逻辑。

导出按需加载（PEP 562）：__all__ 里每个名字在第一次访问时才 import 对应子模块。
延迟不是装饰 —— 两条既有契约都在干净子进程里守着 import 图：
  * tests/test_import_purity.py: import chassis_sim 不得拉起 numba/matplotlib/torch...
  * tests/test_sensors_core.py::test_sensors_package_does_not_import_bridges:
    import chassis_sim.sensors 不得把 chassis_sim.bridges 拉进来。
chassis_sim.envs（环境本体）在模块级就 import bridges，所以顶层不能急切 import 它，
否则任何子模块的 import 都会连带污染 sensors 的纯度。chassis_sim.planning 刻意
**不**导出：它拉 numba，同样破坏 import 纯净性。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from . import gym_registration as _gym_registration  # noqa: F401 - 注册副作用

__version__ = "3.0.1"

#: 公共名 -> 定义它的模块。__all__ 是它的键（外加 __version__），
#: tests/test_api_make.py 锁死两者一致，避免导出名与模块表走散。
_EXPORTS: dict[str, str] = {
    "make": "chassis_sim.api",
    "EnvConfig": "chassis_sim.config",
    "RewardConfig": "chassis_sim.config",
    "load_default_config": "chassis_sim.config",
    "from_config_dir": "chassis_sim.config",
    "ChassisSimEnv": "chassis_sim.envs",
    "GEAR_EPS": "chassis_sim.paths",
    "PathExtractor": "chassis_sim.paths",
    "cartesian_to_polar": "chassis_sim.paths",
    "gear_is_compliant": "chassis_sim.paths",
    "np_angle_wrap": "chassis_sim.paths",
    "segment_is_reverse": "chassis_sim.paths",
    "RewardCalculator": "chassis_sim.reward",
    "SafetyGate": "chassis_sim.safety",
    "ObsSpec": "chassis_sim.obs",
    "ResolvedObsSpec": "chassis_sim.obs",
    "ObsContext": "chassis_sim.obs",
    "Segment": "chassis_sim.obs",
    "Odom": "chassis_sim.obs",
    "PathSegment": "chassis_sim.obs",
    "Goal": "chassis_sim.obs",
    "Act": "chassis_sim.obs",
    "Err": "chassis_sim.obs",
    "GoalSig": "chassis_sim.obs",
    "PathValid": "chassis_sim.obs",
    "PathProgress": "chassis_sim.obs",
    "LidarSegment": "chassis_sim.obs",
    "Custom": "chassis_sim.obs",
    "Lidar": "chassis_sim.sensors",
    "Imu": "chassis_sim.sensors",
    "WheelOdom": "chassis_sim.sensors",
    "Controller": "chassis_sim.controllers",
    "PurePursuitController": "chassis_sim.controllers",
    "DwaController": "chassis_sim.controllers",
    "make_controller": "chassis_sim.controllers",
    "record": "chassis_sim.recording",
    "snapshot": "chassis_sim.recording",
    "evaluate": "chassis_sim.eval",
    "EvalResult": "chassis_sim.eval",
    "Chassis": "chassis_sim.chassis",
    "ChassisCommand": "chassis_sim.chassis",
    "AckermannChassis": "chassis_sim.chassis",
    "DiffDriveChassis": "chassis_sim.chassis",
    "MecanumChassis": "chassis_sim.chassis",
    "CHASSIS_REGISTRY": "chassis_sim.chassis",
    "available_chassis": "chassis_sim.chassis",
    "make_chassis": "chassis_sim.chassis",
    "list_scenarios": "chassis_sim.scenarios",
    "load_scenario": "chassis_sim.scenarios",
}

__all__ = [
    "AckermannChassis",
    "Act",
    "CHASSIS_REGISTRY",
    "Chassis",
    "ChassisCommand",
    "ChassisSimEnv",
    "Controller",
    "Custom",
    "DiffDriveChassis",
    "DwaController",
    "EnvConfig",
    "Err",
    "EvalResult",
    "GEAR_EPS",
    "Goal",
    "GoalSig",
    "Imu",
    "Lidar",
    "LidarSegment",
    "MecanumChassis",
    "ObsContext",
    "ObsSpec",
    "Odom",
    "PathExtractor",
    "PathProgress",
    "PathSegment",
    "PathValid",
    "PurePursuitController",
    "ResolvedObsSpec",
    "RewardCalculator",
    "RewardConfig",
    "SafetyGate",
    "Segment",
    "WheelOdom",
    "__version__",
    "available_chassis",
    "cartesian_to_polar",
    "evaluate",
    "from_config_dir",
    "gear_is_compliant",
    "list_scenarios",
    "load_default_config",
    "load_scenario",
    "make",
    "make_chassis",
    "make_controller",
    "np_angle_wrap",
    "record",
    "segment_is_reverse",
    "snapshot",
]


def __getattr__(name: str) -> Any:
    """首次访问时按 _EXPORTS 加载，随后写回 globals()（此后不再走这里）。"""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(__all__)
