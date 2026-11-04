"""chassis-sim 可组合观测段（0.2.0 观测段核心）。

段顺序即拼接顺序，obs 仍是扁平 float32；spec.describe() 给出 name -> slice。
默认 spec = odom + path + goal + act。env 侧填 ObsContext 的接线在 obs/context.py
（ContextBuilder），本包不 import envs。
"""

from chassis_sim.obs.spec import (
    Act,
    Custom,
    Err,
    Goal,
    GoalSig,
    LidarSegment,
    ObsContext,
    ObsSpec,
    Odom,
    PathProgress,
    PathSegment,
    PathValid,
    ResolvedObsSpec,
    Segment,
)

__all__ = [
    "Act",
    "Custom",
    "Err",
    "Goal",
    "GoalSig",
    "LidarSegment",
    "ObsContext",
    "ObsSpec",
    "Odom",
    "PathSegment",
    "PathProgress",
    "PathValid",
    "ResolvedObsSpec",
    "Segment",
]
