# 迁移指南：0.2.x → 1.0.0

1.0.0 **不提供兼容层**：删掉的名字没有别名、没有 DeprecationWarning，直接 ImportError /
TypeError / AttributeError。本文给一张 旧 → 新 对照表和可跑的替换代码；每个 `python`
块的输出都是真跑出来的。

对照表按“你原来写的什么”排序，右列就是 1.0.0 的写法。

## 1. 对照表

| 旧（0.2.x） | 新（1.0.0） | 备注 |
| --- | --- | --- |
| `ObsLayout(2, 30).as_spec()` | 手写九段 `ObsSpec` | 494 维不再是预设，见第 2 节 |
| `ObsLayout.build(...)` 逐位对拍 | 显式 spec + golden 切片表 | 见第 2 节末尾 |
| `chassis_sim.obs.Path` | `chassis_sim.obs.PathSegment` | 段名仍是 `"path"` |
| `chassis_sim.envs.path_extractor.PathExtractor` | `chassis_sim.paths.PathExtractor` | 同模块还有 `cartesian_to_polar` / `np_angle_wrap` / `segment_is_reverse` / `gear_is_compliant` / `GEAR_EPS` |
| `chassis_sim.envs.path_utils.*` | `chassis_sim.paths.*` | `angle_wrap` / `clip_path_to_bounds` / `subsample_path` |
| `chassis_sim.envs.safety_gate.SafetyGate` | `chassis_sim.safety.SafetyGate` | |
| `chassis_sim.envs.reward.RewardCalculator` | `chassis_sim.reward.RewardCalculator` | |
| `chassis_sim.envs.config.EnvConfig` / `RewardConfig` / `from_config_dir` / `load_default_config` | `chassis_sim.config.*`（顶层也有） | |
| `from chassis_sim.envs import <任何非 ChassisSimEnv 的名字>` | 从对应子模块或顶层取 | `chassis_sim.envs.__all__ == ["ChassisSimEnv"]` |
| `bridge.get_scan()` / `get_lidar_front()` / `get_lidar_rear()` / `scan_to_pointcloud()` | `bridge.get_sensor_reading()` | 唯一入口，返回按传感器名索引的 dict |
| `ChassisSimEnv(..., sensors=[...])` + `info["scan"]` | `cs.make(..., sensors=[...])` + `info[<传感器名>]` | 键就是 `Lidar` / `Imu` / `WheelOdom` 的 `name` |
| `evaluate(..., n_episodes=N)` | `evaluate(..., episodes=N)` | 旧关键字直接 `TypeError` |
| `ChassisSimEnv("ackermann", max_steps=...)` | `cs.make("ackermann", max_steps=...)` | 形状、语义都不变，只是多了一层统一入口 |
| 默认 env 自带雷达 | 默认**一支都不装** | `info["d_min"] == bridge.max_range`，见第 8 节 |
| `EnvConfig.domain_rand_enabled` 默认 `True` | 默认 `False` | 要随机化就显式打开 |

旧名字现在的真实报错（实测）：

```text
n_episodes TypeError: evaluate() got an unexpected keyword argument 'n_episodes'
envs.EnvConfig: ImportError cannot import name 'EnvConfig' from 'chassis_sim.envs'
envs.path_extractor: ModuleNotFoundError No module named 'chassis_sim.envs.path_extractor'
ObsLayout: ImportError cannot import name 'ObsLayout' from 'chassis_sim'
obs.Path: ImportError cannot import name 'Path' from 'chassis_sim.obs'
has get_scan: False | has get_sensor_reading: True
```

## 2. 观测：`ObsLayout(2, 30).as_spec()` → 九段 `ObsSpec`

0.2.x 用 `ObsLayout(action_dim=2, max_path_points=30).as_spec()` 一次性拿到 494 维
布局。1.0.0 把这九段原样保留，但要你自己拼，因为 `LidarSegment` 的宽度取决于你声明的
雷达 —— 这个信息只有调用方知道。段序与切片与 0.2.x 完全一致：

```python
import chassis_sim as cs

spec = cs.ObsSpec(
    (
        cs.LidarSegment("lidar", sensors=("front", "rear")),
        cs.Odom(),
        cs.PathSegment(points=30),
        cs.Goal(),
        cs.Act(action_dim=2),
        cs.Err(),
        cs.GoalSig(),
        cs.PathValid(),
        cs.PathProgress(),
    )
)
resolved = spec.resolved({"front": {"n_rays": 180}, "rear": {"n_rays": 180}})
print("total:", resolved.total_dim)

GOLDEN = (
    ("lidar", 0, 360),
    ("odom", 360, 364),
    ("path", 364, 484),
    ("goal", 484, 486),
    ("act", 486, 488),
    ("err", 488, 490),
    ("goal_sig", 490, 492),
    ("path_valid", 492, 493),
    ("path_progress", 493, 494),
)
actual = tuple(
    (name, span.start, span.stop) for name, span in resolved.describe().items()
)
print("golden match:", actual == GOLDEN)
```

```text
total: 494
golden match: True
```

注意 `resolved()` 的入参：`LidarSegment` 在不知道自己引用几支、每支多少线的时候算不出
宽度，所以要么传入 `{"front": {"n_rays": 180}, "rear": {"n_rays": 180}}` 这样的元数据，
要么在构造时直接写 `ray_counts={"front": 180, "rear": 180}`。真正的环境构造不需要这一步
—— `cs.make(...)` 会从你给的 `sensors=[...]` 里取线数。

### `ObsLayout.build()` 的逐位对拍怎么搬

0.2.x 的测试常写成“`build()` 的第 360:364 切片等于 odom”。1.0.0 把对拍对象换成两样东西：

1. **切片表** —— 上例的 `GOLDEN` 元组对 `resolved().describe()`；
2. **环境级 golden 向量** —— 用同一 seed 跑 `env.reset(seed=0)` 拿到 `obs`，逐段与
   0.2.x 录下来的参考向量比较（`np.array_equal`），这才是“逐位”的那一层。

段名与切片由 `obs_spec_describe()` 发布，测试里不要再写死 `360:364` 这类常量。

## 3. 观测段改名：`obs.Path` → `PathSegment`

```python
import chassis_sim as cs

env = cs.make(
    "ackermann",
    obs=cs.ObsSpec(
        (cs.Odom(), cs.PathSegment(points=30), cs.Goal(), cs.Act(action_dim=2))
    ),
)
print(env.obs_spec_describe())
```

```text
{'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 128, None)}
```

改名只影响类名，`describe()` 里的键还是 `"path"` —— 依赖旧键的下游代码不用改，依赖旧
类名的 import 必须改。

## 4. 深路径公共化：`chassis_sim.envs.*` → `chassis_sim.{paths,safety,reward,config}`

```python
from chassis_sim.config import (
    EnvConfig,
    RewardConfig,
    from_config_dir,
    load_default_config,
)
from chassis_sim.paths import (
    GEAR_EPS,
    PathExtractor,
    cartesian_to_polar,
    gear_is_compliant,
    np_angle_wrap,
    segment_is_reverse,
)
from chassis_sim.reward import RewardCalculator
from chassis_sim.safety import SafetyGate

extractor = PathExtractor(lookahead=8.0, target_count=30)
gate = SafetyGate(wheelbase=0.145)
calc = RewardCalculator(weights={"progress": 1.0})
print(type(extractor).__name__, type(gate).__name__, type(calc).__name__)
print("GEAR_EPS:", GEAR_EPS)
print(
    "np_angle_wrap(1.5 * 3.141592653589793):",
    round(float(np_angle_wrap(1.5 * 3.141592653589793)), 6),
)
print(
    "helpers:",
    cartesian_to_polar.__name__,
    gear_is_compliant.__name__,
    segment_is_reverse.__name__,
)
```

```text
PathExtractor SafetyGate RewardCalculator
GEAR_EPS: 0.01
np_angle_wrap(1.5 * 3.141592653589793): -1.570796
helpers: cartesian_to_polar gear_is_compliant segment_is_reverse
```

四个模块的逻辑没改，只是从 `chassis_sim.envs` 的私有角落搬到了稳定名字下。
`chassis_sim.envs` 现在只导出 `ChassisSimEnv` 一个名字，其余都别从这里 import。
这几个名字顶层也 re-export（`cs.PathExtractor`、`cs.SafetyGate`、
`cs.RewardCalculator`、`cs.EnvConfig` 等），脚本里用哪个都行。

## 5. 激光：四个访问器 → `get_sensor_reading()`

旧写法（`get_scan()` / `get_lidar_front()` / `get_lidar_rear()` /
`scan_to_pointcloud()`）全部删除。1.0.0 只有一个入口，返回按传感器名索引的 dict：

```python
import numpy as np

import chassis_sim as cs

env = cs.make(
    "ackermann",
    sensors=[cs.Lidar("front", mount=(0.145, 0, 0), n_rays=90)],
    seed=0,
)
env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
reading = env.unwrapped.bridge.get_sensor_reading()
print("readings:", sorted(reading))
print(
    "scan:",
    reading["front"]["scan"].shape,
    "valid:",
    reading["front"]["scan_meta"]["valid"],
)
print("info carries the same dict:", sorted(info["front"]))
```

```text
readings: ['front']
scan: (90,) valid: True
info carries the same dict: ['scan', 'scan_meta']
```

两个连带变化：传感器读数在 `info` 里的键是**传感器名**（旧版固定为
`info["scan"]` / `info["imu"]` / `info["wheel"]`）；不配传感器时 `info` 里没有任何
传感器键。点云转换不在包里 —— `scan_to_pointcloud()` 是使用者侧的活。

## 6. 评测：`n_episodes=` → `episodes=`

```python
import chassis_sim as cs

env = cs.make("ackermann", max_steps=200)
res = cs.evaluate(env, episodes=3)
print("episodes:", res.n, "successes:", res.successes, "mean_steps:", res.mean_steps)
```

```text
episodes: 3 successes: 3 mean_steps: 65.0
```

顺带两件事：`episodes` 是 keyword-only（没有位置参数版本）；`evaluate()` 允许省略
控制器 —— 省略时按 `env.unwrapped.chassis` 自动绑 `PurePursuitController`，要换控制器就
传 `cs.make_controller("dwa", env=env)` 之类。表格/JSON 输出用
`res.markdown()`（`to_markdown()` 的别名）。

## 7. 构造：`ChassisSimEnv(...)` → `cs.make(...)`

旧（仍可用，但不再是文档推荐的写法）：

```python
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.sensors import Lidar

env = ChassisSimEnv("ackermann", max_steps=500, sensors=[Lidar("front", n_rays=180)])
print("obs:", env.observation_space.shape)
```

新：

```python
import chassis_sim as cs

env = cs.make("ackermann", max_steps=500, sensors=[cs.Lidar("front", n_rays=180)])
```

`cs.make()` 的第一个参数 positional-only，只接受车型名或 `EnvConfig` 实例；
`cs.make("diff_drive", cfg)` 这种两个位置参数的写法是 `TypeError`，要写成
`cs.make(cfg, ...)`。`make()` 的优先级固定为 **kwargs > cfg 字段 > 库默认**，
`scenario=` 只把场景翻译进 `env.default_reset_options`，**不会**自动 `reset()`。

## 8. 默认行为差异：默认环境不再装雷达

这是最容易静默踩到的一条。旧版默认拼一支雷达，于是 `info["d_min"]` 是真实测量的最近
障碍距离。1.0.0 默认一支都不装：

```python
import numpy as np

import chassis_sim as cs

plain = cs.make("ackermann", max_steps=5)
plain.reset(seed=0)
obs, reward, terminated, truncated, info = plain.step(np.zeros(2, dtype=np.float32))
print("d_min:", info["d_min"], "max_range:", plain.unwrapped.bridge.max_range)

sensed = cs.make("ackermann", max_steps=5, sensors=[cs.Lidar("front", n_rays=180)])
sensed.reset(seed=0)
obs, reward, terminated, truncated, info = sensed.step(np.zeros(2, dtype=np.float32))
print("d_min:", round(float(info["d_min"]), 4))
```

```text
d_min: 10.0 max_range: 10.0
d_min: 5.0
```

后果：不装雷达时 reward 的 obstacle 项退化为常数，`d_min` 也不再是测量值。要真实净空就
显式声明 `sensors=[cs.Lidar(...)]`。

## 9. 迁移 checklist

```bash
# 1) 观测预设与旧布局
grep -rn "ObsLayout\|ObservationBuilder\|compute_obs_dims\|BASE_RAW_OBS_DIM" .
# 2) 观测段改名
grep -rn "obs\.Path\b\|from chassis_sim.obs import Path\b" .
# 3) 激光旧访问器
grep -rn "get_scan\|get_lidar_front\|get_lidar_rear\|scan_to_pointcloud" .
# 4) 旧深路径
grep -rn "chassis_sim\.envs\.\(path_extractor\|path_utils\|safety_gate\|reward\|config\)" .
# 5) 旧评测关键字
grep -rn "n_episodes" .
# 6) 旧传感器 info 键
grep -rn 'info\["scan"\]\|info\["imu"\]\|info\["wheel"\]' .
```

六条检索全部清零，剩下的就是 `ChassisSimEnv(...)` → `cs.make(...)` 这类风格迁移。
