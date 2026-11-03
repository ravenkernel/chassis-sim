"""包内自带的 env 配置 YAML（package data，随 wheel 发布）。

本目录只放数据文件：env.yaml / reward.yaml / curriculum.yaml。
读取逻辑与 dataclass 在 chassis_sim.envs.config；这里只做转发，使
`from chassis_sim.config import load_default_config` 也成立。
"""

from chassis_sim.envs.config import (
    EnvConfig,
    RewardConfig,
    from_config_dir,
    load_default_config,
)

__all__ = ["EnvConfig", "RewardConfig", "from_config_dir", "load_default_config"]
