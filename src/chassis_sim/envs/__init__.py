"""`chassis_sim.envs` 环境实现层的门面：只导出 `ChassisSimEnv`。

1.0.0 起公共 API 全部在包顶层（`import chassis_sim`）或下述稳定子模块：

    chassis_sim.paths      路径工具 / 档位几何判定 / PathExtractor
    chassis_sim.safety     SafetyGate
    chassis_sim.reward     RewardCalculator
    chassis_sim.obs        观测 spec（ObsSpec / 段 / ContextBuilder）

本包内的实现模块（`chassis_env` / `privileged` / `dynamics`）保留在原处，
但它们不再是公共入口 —— 除 `ChassisSimEnv` 外这里不 re-export 任何名字。
"""

from chassis_sim.envs.chassis_env import ChassisSimEnv

__all__ = ["ChassisSimEnv"]
