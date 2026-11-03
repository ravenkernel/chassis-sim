"""chassis-sim：三种底盘的 Gymnasium 环境。

import 本包即注册三个 id（ChassisSim-Ackermann-v0 / ChassisSim-DiffDrive-v0 /
ChassisSim-Mecanum-v0）。注册逻辑在 gym_registration.py，只依赖 gymnasium：
环境本体按 entry_point 字符串延迟 import，不拉起 numba / matplotlib / torch。
"""

from . import gym_registration as _gym_registration  # noqa: F401 - 注册副作用

__version__ = "0.1.4"

__all__ = ["__version__"]
