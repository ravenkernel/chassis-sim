"""配置层：EnvConfig / RewardConfig 与包内自带 YAML（package data，随 wheel 发布）。

本目录有两个角色：
  · schema.py —— 配置的唯一家：EnvConfig / RewardConfig 两个 dataclass、
    from_config_dir() / load_default_config()，以及 YAML 快照
    （EnvConfig.to_yaml / EnvConfig.from_yaml）与三级优先级解析 _resolve()。
  · env.yaml / reward.yaml / curriculum.yaml —— 数值的唯一权威，随 wheel 发布；
    load_default_config() 用 importlib.resources 定位它们，与 CWD 无关。

这里只做转发（不写逻辑），保证两个等价写法都成立：
    from chassis_sim.config import EnvConfig
    from chassis_sim.config.schema import EnvConfig
"""

from chassis_sim.config.schema import (
    EnvConfig,
    RewardConfig,
    from_config_dir,
    load_default_config,
)

__all__ = ["EnvConfig", "RewardConfig", "from_config_dir", "load_default_config"]
