"""chassis-sim 的 env 数据层（从主仓 drl-sim/envs/ 移植，运行时只依赖 numpy）。

模块划分与主仓一致：
    observation.py    ObsLayout / ObservationBuilder（raw 观测拼接，无 encoder）
    reward.py         RewardCalculator（稠密 + 终止奖励，含失败补偿与退款上限）
    path_extractor.py 世界系路径 → 车体系定向重采样
    path_utils.py     路径/角度工具
    safety_gate.py    安全门（急停/限速）
    privileged.py     privileged 观测量化（n_bins 扇形）
    config.py         EnvConfig / RewardConfig + load_default_config()
    chassis_env.py    ChassisSimEnv（车型参数化的 gymnasium Env）

与主仓的结构差异：ObsLayout 现在把 prev_action 的宽度作为构造参数
（`ObsLayout(action_dim=2)`），使非 ackermann 底盘不再被写死的 2 维绑死；
`ObsLayout(2)` 的 9 段切片与总维 494 与主仓逐位相同（tests/test_envs_observation.py 锁死）。

0.2.0 起观测由可组合的 ObsSpec 决定（见 chassis_sim.obs）：
  · 默认 = ObsSpec.default(...) 的**最小集** odom+path+goal+act ⇒ 128/128/129 维；
  · 494（含 360 维雷达）不再内置，要就走 `obs_spec=ObsLayout(2, 30).as_spec()`；
  · ObsLayout / ObservationBuilder 保留原行为，作为逐位对拍的金标准。
"""

from chassis_sim.envs.chassis_env import ChassisSimEnv
from chassis_sim.envs.config import (
    EnvConfig,
    RewardConfig,
    from_config_dir,
    load_default_config,
)
from chassis_sim.envs.observation import (
    BASE_RAW_OBS_DIM,
    ObservationBuilder,
    ObsLayout,
    compute_obs_dims,
)
from chassis_sim.envs.path_extractor import PathExtractor
from chassis_sim.envs.reward import RewardCalculator

__all__ = [
    "BASE_RAW_OBS_DIM",
    "ChassisSimEnv",
    "EnvConfig",
    "ObsLayout",
    "ObservationBuilder",
    "PathExtractor",
    "RewardCalculator",
    "RewardConfig",
    "compute_obs_dims",
    "from_config_dir",
    "load_default_config",
]
