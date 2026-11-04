# chassis-sim

[English](https://github.com/zhao6663/chassis-sim/blob/main/README.md) | 简体中文

[![CI](https://github.com/zhao6663/chassis-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/zhao6663/chassis-sim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/zhao6663/chassis-sim)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**三种底盘的小型 Gymnasium 基准** —— 阿克曼转向、差速驱动、麦轮 —— 归一化动作、
纯 NumPy 仿真器、可组合的观测 spec、声明式传感器、参考控制器，以及一套
“建环境 / 评测 / 录制”各一行就够的公共 API。

版本 1.0.0。Python 3.10+。

## 概览

每个环境都暴露归一化动作空间 `Box(-1, 1, (d,), float32)`。底盘负责把归一化动作
翻译成物理量；环境从不要求使用者提供电压、轮端力矩或原始转向角。

| Gymnasium id | 底盘 | 动作维度 | 动作布局 | 默认观测维度 | 默认配置下的物理限值 |
| --- | --- | ---: | --- | ---: | --- |
| `ChassisSim-Ackermann-v0` | 阿克曼 | 2 | `[steer, speed]` | 128 | `v ∈ [-0.6, 1.2] m/s`、`δ ∈ [-0.526, 0.526] rad`、`|ω| ≤ 4.805 rad/s` |
| `ChassisSim-DiffDrive-v0` | 差速驱动 | 2 | `[speed, yaw]` | 128 | `v ∈ [-1.2, 1.2] m/s`、`|ω| ≤ 1.0 rad/s` |
| `ChassisSim-Mecanum-v0` | 麦轮 | 3 | `[vx, vy, yaw]` | 129 | `v_x ∈ [-1.2, 1.2] m/s`、`|vy| ≤ 0.5 m/s`、`|ω| ≤ 1.0 rad/s` |

上表就是默认 `EnvConfig` 下 `env.chassis.describe()["limits"]` 报出的限值；环境内部
不再重复保存一份。换一个 `EnvConfig`（或换一个 `Chassis` 实例）会同时改变物理映射
与动作语义：阿克曼上 `delta_max` 与 `curvature_action` 决定第一维动作是转向角还是
曲率。

默认观测是四段 spec `ObsSpec.default(action_dim, points=30)` —— 里程计、参考路径、
目标点、上一步归一化动作 —— 所以在两个 2 维动作的底盘上是 128 维，麦轮是 129 维。
0.1.x 那个 494 维向量不再是默认值，1.0.0 也不内置 494 预设：九段 `ObsSpec` 由使用
者自己写，见[观测与动作契约](#观测与动作契约)。

## 安装

1.0.0 已上 PyPI：

```bash
pip install "chassis-sim[viz]"
pip install "chassis-sim[viz]==1.0.0"   # 锁定版本
```

只要核心依赖就去掉 `[viz]`；Hybrid-A* 地图生成器在 `[planning]` 里。想按 tag 锁定
而不是走 PyPI，每个 tag 的 [GitHub release](https://github.com/zhao6663/chassis-sim/releases)
页面都挂着 wheel。0.2.x 是迁移案例：它的默认观测、传感器访问器和若干模块路径都与
1.0.0 不同 —— 对照表见 [docs/migration-1.0.md](docs/migration-1.0.md)。

从源码安装（含开发工具）：

```bash
git clone https://github.com/zhao6663/chassis-sim.git
cd chassis-sim
pip install -e ".[dev]"
```

### 可选 extras

| Extra | 装什么 | 用途 |
| --- | --- | --- |
| `planning` | `numba>=0.56` | planning 场景的 Hybrid-A* 地图生成 |
| `viz` | `matplotlib>=3.7`、`rich>=13.0`、`imageio>=2.28` | 渲染、GIF 导出、终端富文本 |
| `dev` | `pytest>=7.0`、`ruff>=0.4`、`mypy` | 测试、lint、类型检查 |

核心依赖只有 `numpy`、`gymnasium`、`pyyaml`。`matplotlib` 只给渲染用，`imageio`
只给 GIF 导出用，两者都在 `viz` 里 —— `cs.record()`、`cs.snapshot()` 和
`examples/gallery.py` 依赖的就是它。

1.0.0 只内置一个仿真后端：纯 NumPy 的 `SimpleBridge`。不需要任何外部仿真器，也没
有对应的 optional extra：ROS 2 / Gazebo 一类后端不在本包范围内。
`PhysicalCommandBridge` 是把自己的物理引擎接进同一套观测/奖励流水线的正式接缝。

## 快速开始

一次 import、一个构造函数、一个评测器：

```python
import chassis_sim as cs

env = cs.make("ackermann", seed=0)
print("obs:", env.observation_space.shape)
print("describe:", env.obs_spec_describe())

ctrl = cs.make_controller("pp", env=env)
res = cs.evaluate(env, ctrl, episodes=20)
print(
    "success_rate:", f"{res.success_rate:.0%}", "mean_steps:", f"{res.mean_steps:.1f}"
)
print(res.markdown().splitlines()[2])

rec = cs.record(env, ctrl, out="demo.gif", every=4, fps=12)
print("gif:", rec.path, "frames:", rec.frames, "steps:", rec.steps)

auto = cs.evaluate(cs.make("ackermann"), episodes=3)
print("controller omitted ->", auto.controller, "|", auto.successes, "of", auto.n)
```

输出：

```text
obs: (128,)
describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 128, None)}
success_rate: 100% mean_steps: 65.0
| controller | ackermann |
gif: demo.gif frames: 17 steps: 65
controller omitted -> PurePursuitController | 3 of 3
```

`cs.make()` 的解析顺序固定为 **kwargs > `EnvConfig` 字段 > 库默认**，因此
`cs.make(cfg, v_max=0.8)` 覆盖 `cfg.v_max` 且不改动 `cfg` 本身。第一个参数是
positional-only，只接受底盘名或 `EnvConfig` 实例；其余一律关键词参数。

`cs.evaluate(env, controller=None, *, episodes=...)` 跑控制器并返回 `EvalResult`。
省略控制器是受支持的：评测器会把 `PurePursuitController` 绑到
`env.unwrapped.chassis` 上再跑，所以最短版本连控制器对象都不需要。想用别的控制器就
显式传 —— `cs.make_controller("pp", env=env)` 与
`cs.make_controller("dwa", env=env)` 返回的控制器的已经按该底盘标定好。

`cs.record(env, controller, out="demo.gif", every=4, fps=12)` 把一个回合重放成 GIF
（65 步的回合、`every=4` ⇒ 17 帧），返回 `RecordingResult`，字段有 `path`、
`frames`、`steps`、`total_reward`、`terminated`、`truncated`。
`cs.snapshot(env, "hero.png")` 把当前状态存成一张静图。

## 图库

十一份素材，一条命令在场景 seed 0 下全部重建：

```bash
python examples/gallery.py --out docs/media
```

五段 GIF 是真实闭环跑出来的 12 fps 回放：控制器从 seed-0 的 reset 状态出发驱动环境，
录下来的动作序列再从同一初态重放，因此每段都可复现，也都真的到达了目标。六张静图是
场景的初始状态（`reset` 后第 0 步，底盘 `ackermann`，不挂控制器），画面里带障碍场、
参考路径、车体，以及角落里的底盘名与场景名。每张图链到 `docs/media/` 下的原尺寸文件。

### 底盘运动学 —— 一条参考路径，一个控制器

三台底盘跑同一条 `obstacles_sparse` 路径、用同一个 `pure_pursuit` 控制器，唯一的
变量就是运动学。

<table>
<tbody>
<tr>
<td align="center" width="33%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_ackermann.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_ackermann.gif" width="250" alt="阿克曼底盘跟踪 obstacles_sparse 路径"></a><br><code>chassis_ackermann.gif</code><br><em>obstacles_sparse · ackermann · pure_pursuit</em></td>
<td align="center" width="33%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_diff_drive.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_diff_drive.gif" width="250" alt="差速底盘跟踪同一条 obstacles_sparse 路径"></a><br><code>chassis_diff_drive.gif</code><br><em>obstacles_sparse · diff_drive · pure_pursuit</em></td>
<td align="center" width="33%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_mecanum.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_mecanum.gif" width="250" alt="麦轮底盘跟踪同一条 obstacles_sparse 路径"></a><br><code>chassis_mecanum.gif</code><br><em>obstacles_sparse · mecanum · pure_pursuit</em></td>
</tr>
</tbody>
</table>

前轮转向受最小转弯半径约束，阿克曼车过弯必须甩出去一段。差速底盘没有横向自由度，
转向全靠左右轮速差，在同一条路径上读起来就是更紧的一次折角。麦轮多一维正交速度分量，
直接在车体上做横向修正，因此最贴合参考线。

### 控制器 —— 一台底盘，一个场景

两段 GIF 都是 `ackermann` 底盘跑 `dense_static`，差别只在控制器。

<table>
<tbody>
<tr>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/controller_pure_pursuit.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/controller_pure_pursuit.gif" width="400" alt="纯追踪控制器跑 dense_static 路径"></a><br><code>controller_pure_pursuit.gif</code><br><em>dense_static · ackermann · pure_pursuit</em></td>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/controller_dwa.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/controller_dwa.gif" width="400" alt="DWA 控制器跑 dense_static 路径"></a><br><code>controller_dwa.gif</code><br><em>dense_static · ackermann · dwa</em></td>
</tr>
</tbody>
</table>

一段前视圆弧让纯追踪跟得平顺，代价是切弯、车体离障碍更近。速度窗采样加上障碍代价让
DWA 走更保守的一条线，用更少的步数收工。

### 场景 —— 俯视图

<table>
<tbody>
<tr>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_straight.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_straight.png" width="400" alt="straight：空场景，只有参考路径与车体轮廓"></a><br><code>scene_straight.png</code><br><em>straight · seed 0 · reset 后第 0 步</em></td>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_corridor.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_corridor.png" width="400" alt="corridor：两堵长墙夹出可行带"></a><br><code>scene_corridor.png</code><br><em>corridor · seed 0 · reset 后第 0 步</em></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_dense_static.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_dense_static.png" width="400" alt="dense_static：14 个障碍贴参考路径排布"></a><br><code>scene_dense_static.png</code><br><em>dense_static · seed 0 · reset 后第 0 步</em></td>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_obstacles_dense.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_obstacles_dense.png" width="400" alt="obstacles_dense：Hybrid-A* 地图生成器产物"></a><br><code>scene_obstacles_dense.png</code><br><em>obstacles_dense · seed 0 · reset 后第 0 步</em></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_dynamic_crossing.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_dynamic_crossing.png" width="400" alt="dynamic_crossing：动态圆在 t=0 时的位姿"></a><br><code>scene_dynamic_crossing.png</code><br><em>dynamic_crossing · seed 0 · reset 后第 0 步</em></td>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_obstacles_sparse.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_obstacles_sparse.png" width="400" alt="obstacles_sparse：Hybrid-A* 地图生成器产物"></a><br><code>scene_obstacles_sparse.png</code><br><em>obstacles_sparse · seed 0 · reset 后第 0 步</em></td>
</tr>
</tbody>
</table>

`straight` 是干净基线：无阻碍，y = 5 处一条 4 m 参考线。`corridor` 把可行域压到两堵
长矩形之间的 0.6 m 带里，所以贴墙余量一眼可读。`dense_static` 是手写的 14 个圆/矩形
贴着参考路径排布，不需要规划器。`obstacles_dense` 与 `obstacles_sparse` 来自
Hybrid-A* 地图生成器，障碍场按 seed 现场生成、不存 YAML：dense 抽 9–15 个贴着路径的
障碍，sparse 在开阔区抽 2–5 个。`dynamic_crossing` 有一个 0.25 m 的圆以 0.3 m/s
沿 20 s 乒乓周期横穿路径，图中画的是它 t = 0 的位姿。

## 观测与动作契约

观测是一个 `float32` 向量，由一段一段有序拼装而成。默认那段列表就是
`ObsSpec.default(action_dim, points=30)`：

`ObsSpec((Odom(), PathSegment(points=30), Goal(), Act(action_dim=action_dim)))` ——
里程计、参考路径、目标点、上一步归一化动作，顺序照写。

| 段 | 构造函数 | 宽度 | 含义 |
| --- | --- | ---: | --- |
| `Odom` | `Odom()` | 4 | 车体系 `[dx*10, dy*10, v, ω]` |
| `PathSegment` | `PathSegment(points=30)` | `4 * points` = 120 | 车体系参考路径，每点 `[x, y, ψ, s]` |
| `Goal` | `Goal()` | 2 | 车体系目标点 `[x, y]` |
| `Act` | `Act(action_dim=d)` | `d` | 上一步归一化动作 |

也就是 `4 + 120 + 2 + d`：阿克曼与差速是 **128**，麦轮是 **129**（动作多一维）。
`obs_spec_describe()` 返回权威切片表，任何下游都不该硬编码偏移：

```python
import chassis_sim as cs

env = cs.make("ackermann")
print(
    "default obs:", env.observation_space.shape, "dtype:", env.observation_space.dtype
)
print("describe:", env.obs_spec_describe())

mecanum = cs.make("mecanum")
print("mecanum obs:", mecanum.observation_space.shape)
print("describe:", mecanum.obs_spec_describe())
```

```text
default obs: (128,) dtype: float32
describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 128, None)}
mecanum obs: (129,)
describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 129, None)}
```

### 显式的 494 维预设

0.2.x 的 494 维布局是一个九段 spec（`lidar 360 + odom 4 + path 120 + goal 2 +
act 2 + err 2 + goal_sig 2 + path_valid 1 + path_progress 1`，`action_dim=2` 时）。
1.0.0 把这些段一个不落地留着，但不内置由它们拼成的预设 —— 因为 LiDAR 段只有在你
显式声明了它要读的传感器之后才有意义。自己把 spec 写出来：

```python
import numpy as np

import chassis_sim as cs

env = cs.make(
    "ackermann",
    scenario="obstacles_dense",
    sensors=[
        cs.Lidar("front", mount=(0.145, 0, 0), n_rays=180),
        cs.Lidar("rear", mount=(-0.155, 0, np.pi), n_rays=180),
    ],
    obs=cs.ObsSpec(
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
    ),
)
print("obs:", env.observation_space.shape)
for name, span in env.obs_spec_describe().items():
    print(f"  {name:14s} {span.start:3d}-{span.stop:3d}")
```

```text
obs: (494,)
  lidar            0-360
  odom           360-364
  path           364-484
  goal           484-486
  act            486-488
  err            488-490
  goal_sig       490-492
  path_valid     492-493
  path_progress  493-494
```

段序 = 构造顺序，因此上面那张切片表就是契约：`lidar 0-360`（两路 180 线扫描首尾相接）、
`odom 360-364`、`path 364-484`、`goal 484-486`、`act 486-488`、`err 488-490`、
`goal_sig 490-492`、`path_valid 492-493`、`path_progress 493-494`。`obs=` 是
`cs.make()` 里 `obs_spec=` 的正式别名。

### 物理指令桥接

第三方物理后端（ROS 2、Gazebo，或你自己的积分器）吃的是**物理**指令，
不是归一化动作。`PhysicalCommandBridge` 用 `chassis.physical_action()` 做映射（绝不
自己调 `step()` —— 执行器滞后归后端所有），传感器原样转发。物理向量顺序就是底盘自己
的顺序：

| 底盘 | 物理向量 |
| --- | --- |
| `ackermann` | `[delta (rad), v (m/s)]` |
| `diff_drive` | `[v (m/s), omega (rad/s)]` |
| `mecanum` | `[v (m/s), vy (m/s), omega (rad/s)]` |

`bridge.step()` 收的仍然是**归一化**动作，内部把 `chassis.physical_action(action)`
交给 `inner.step()`；`bridge.to_physical(action)` 把这个映射暴露成纯函数，所以同一个
归一化动作永远得到同一个物理向量，与执行器状态无关。执行器滞后必须**只施加一次**，
由内层物理仿真器施加：适配器从不推进底盘状态，也从不调 `chassis.reset_state()`。

```python
from chassis_sim.bridges import PhysicalCommandBridge
from chassis_sim.chassis import AckermannChassis


class InnerBridge:
    """stands in for an external simulator that speaks physical commands"""

    def __init__(self):
        self.commands = []

    def step(self, commands):
        self.commands.append([float(value) for value in commands])


chassis = AckermannChassis()
inner = InnerBridge()
bridge = PhysicalCommandBridge(inner, chassis)
print("physical_action([1.0, 0.5]):", chassis.physical_action([1.0, 0.5]))
bridge.step([0.3, 0.6])
print("inner received:", inner.commands)
```

```text
physical_action([1.0, 0.5]): [0.52558379 0.5       ]
inner received: [[0.1722752169535286, 0.6]]
```

## 传感器

传感器是一串不可变的 spec：想要哪几支就写哪几支，交给 `cs.make(sensors=[...], seed=...)`。
**默认环境一支都不装。**

| Spec | 字段（默认值） |
| --- | --- |
| `Lidar` | `name`、`mount=(x, y, yaw)` 默认 `(0, 0, 0)`、`fov=π`、`n_rays=180`、`max_range=10.0`、`range_min=0.05`、`noise_std=0.0`、`quant_step=0.0`、`latency_steps=0`、`dropout_prob=0.0` |
| `Imu` | `name`、`noise_std=0.0` |
| `WheelOdom` | `name`、`track_width=None`（回落到底盘轮距）、`noise_std=0.0` |

读数经 `reset()` / `step()` 的 `info` 曝光，**键就是传感器名**，本身绝不进观测：

| 键 | 值 |
| --- | --- |
| `info[<雷达名>]` | `{"scan": (n_rays,) float32 或 None, "scan_meta": {...}}` |
| `info[<IMU 名>]` | `{"ax", "ay", "yaw_rate", "yaw", "t"}` |
| `info[<轮速计名>]` | `{"v_left", "v_right", "v", "t"}` |

`bridge.get_sensor_reading()` 是唯一的原始入口，返回的就是 `info` 各键取自的那份
字典。1.0.0 没有按传感器类型分开的便利访问器：一个方法覆盖所有传感器，名字和参数都
不影响调用。不配传感器时 `info` 里一个传感器键都没有。

```python
import numpy as np

import chassis_sim as cs

plain = cs.make("ackermann")
plain.reset(seed=0)
obs, reward, terminated, truncated, info = plain.step(np.zeros(2, dtype=np.float32))
print("default info keys:", sorted(info))
print("d_min:", info["d_min"], "bridge.max_range:", plain.unwrapped.bridge.max_range)

sensed = cs.make(
    "ackermann",
    sensors=[
        cs.Lidar("front", mount=(0.145, 0, 0), n_rays=180, noise_std=0.01),
        cs.Imu("imu", noise_std=0.02),
        cs.WheelOdom("wheel", noise_std=0.01),
    ],
    seed=7,
)
sensed.reset(seed=0)
obs, reward, terminated, truncated, info = sensed.step(np.zeros(2, dtype=np.float32))
print("sensor info keys:", sorted(info))
print("front scan:", info["front"]["scan"].shape, info["front"]["scan"].dtype)
print(
    "front meta:",
    {k: info["front"]["scan_meta"][k] for k in ("valid", "dropped", "range_max")},
)
print("imu:", {k: round(float(v), 6) for k, v in info["imu"].items()})
print("wheel:", {k: round(float(v), 6) for k, v in info["wheel"].items()})
print("d_min:", round(float(info["d_min"]), 4))
print(
    "bridge.get_sensor_reading:", sorted(sensed.unwrapped.bridge.get_sensor_reading())
)
```

```text
default info keys: ['collision', 'cte', 'd_min', 'goal_reached', 'map_fallback', 'mapped_action', 'path_deviation', 'path_deviation_cte', 'path_dir', 'path_dir_gate', 'progress', 'reverse_triggered', 'reward_components', 'safety_brake', 'step', 'stuck', 'termination_reason', 'timeout']
d_min: 10.0 bridge.max_range: 10.0
sensor info keys: ['collision', 'cte', 'd_min', 'front', 'goal_reached', 'imu', 'map_fallback', 'mapped_action', 'path_deviation', 'path_deviation_cte', 'path_dir', 'path_dir_gate', 'progress', 'reverse_triggered', 'reward_components', 'safety_brake', 'step', 'stuck', 'termination_reason', 'timeout', 'wheel']
front scan: (180,) float32
front meta: {'valid': True, 'dropped': False, 'range_max': 10.0}
imu: {'ax': 0.019794, 'ay': -0.005537, 'yaw_rate': 0.033798, 'yaw': 0.0, 't': 0.1}
wheel: {'v_left': -0.010802, 'v_right': -0.005745, 'v': -0.008273, 't': 0.1}
d_min: 4.9867
bridge.get_sensor_reading: ['front', 'imu', 'wheel']
```

两个后果值得读两遍。第一，在声明雷达之前 `info["d_min"]` 就等于 `10.0` —— 正好是
`bridge.max_range` —— 因为没有任何东西量过障碍距离：这种配置下 reward 的 obstacle
项退化为常数。第二，除非有 `LidarSegment` 显式点名，开不开传感器观测逐位相同。

### 缺帧与 LiDAR 观测段

`LidarSegment` 观测段把被点名传感器的扫描写进观测向量；产生那份扫描的是
`chassis_sim.sensors.Lidar` 这个传感器 spec。`reset()` 之后桥会立刻产出一帧真实
数据，reset 观测带的就是那一帧。

缺帧时 —— 延迟帧还在延迟环里，或某帧被丢弃 —— 该段**保持上一帧有效扫描**
（零阶保持）。如果一帧有效数据都还没交付过，该段填 `max_range`。`env.reset()` 会
清空保持缓存。观测本身不带有效位：判断某帧是否缺失的唯一依据仍然只有 `info` 里的
`scan_meta["valid"]` 与 `scan_meta["dropped"]`。

LiDAR 处理顺序是固定的：裁剪到 `[range_min, range_max]`、加高斯噪声、再裁剪、量化、
转 `float32`、进延迟环，最后在交付时应用丢帧。因此延迟帧带的是它被采集那一步的噪声，
被丢掉的正是本该交付的那一帧。每支传感器都有独立的随机流，由
`(sensor_seed, name)` 派生，增删一支不会挪动其他支的噪声。

## 控制器

| 类 | 角色 | 关键默认值 |
| --- | --- | --- |
| `PurePursuitController` | 几何路径跟踪 | `lookahead=0.6`、`adaptive=False`、`lookahead_gain=0.8` |
| `DwaController` | 采样式局部规划 | `dt=0.1`、`horizon=10`、`gear_mode="off"` |

构造只有一个正式写法：`make_controller()`。

```python
import chassis_sim as cs

try:
    cs.PurePursuitController()
except TypeError as exc:
    print("TypeError:", exc)

env = cs.make("ackermann")
pp = cs.make_controller("pp", env=env)
dwa = cs.make_controller("dwa", env=env)
print("pp:", type(pp).__name__, "dwa:", type(dwa).__name__)

same = cs.make_controller("pp", chassis=env.unwrapped.chassis)
print("explicit chassis:", type(same).__name__)

result = cs.evaluate(env, pp, episodes=5)
print("with explicit controller:", result.n, f"{result.success_rate:.0%}")

auto = cs.evaluate(cs.make("ackermann"), episodes=5)
print("controller omitted:", auto.controller, f"{auto.success_rate:.0%}")
```

```text
TypeError: Controller 需要 chassis= 或 kind= 之一来确定车型
pp: PurePursuitController dwa: DwaController
explicit chassis: PurePursuitController
with explicit controller: 5 100%
controller omitted: PurePursuitController 100%
```

裸写 `PurePursuitController()` 抛 `TypeError` 是刻意的：控制器按某台底盘的限值归一化
输出，环境按 `env.unwrapped.chassis` 反解。拿错标定就会把每条指令静默缩放（用
`make_chassis("diff_drive")` 构造的控制器去驱动阿克曼 arena：请求 0.5 m/s，实跑
0.6 m/s），所以本包任何地方都不给默认底盘。`evaluate()` 会比对两侧限值，不一致时
**在开跑之前**抛 `ValueError`；`allow_mismatch=True` 是显式豁免。

`DwaController.gear_mode="off"` 是刻意的默认。该模式下候选轨迹覆盖整个
`[v_min, v_max]` 区间，代价函数里没有挡位（前进/倒车）项，倒车交给弧长前视参考处理。
`gear_mode="auto"` 是可选的消融项：它从路径段推断挡位、把候选截到单一半轴、按运动
方向打朝向分，并且需要最低速度才不至于刹停；它对 `heading_lookahead` 敏感，所以不做
默认。

### 自定义控制器

`evaluate()` 不 import 任何具体控制器，只要求这套鸭子类型协议：

| 成员 | 必需 | 契约 |
| --- | --- | --- |
| `reset()` | 是 | 复位内部状态 |
| `set_path(path_world)` | 是 | `(N, 3)` 世界系 `[x, y, θ]`；`θ` 是车头朝向 |
| `act(observation)` | 是 | 返回归一化动作 |
| `to_chassis_action(v, omega, vy=0.0)` | 否 | 内置控制器用的辅助方法 |

传给 `act()` 的观测字典有 `pose`（3 个数）、可选 `obstacles`、可选 `dt`。
`Controller` 基类已经实现好 `to_chassis_action()`。`examples/eval_matrix.py` 里有个
只实现 `reset()` / `set_path()` / `act()` 的 `RandomController`，就是最小集成样例。
本包不附带 MPC 控制器。

## 评测

`evaluate(env, controller=None, *, episodes, seed=0, max_steps=None,
reset_options=None, labels=None, allow_mismatch=False)` 让控制器跑 `episodes` 个
回合并返回 `EvalResult`。回合数是 keyword-only 参数，名字就叫 `episodes`。

```python
import chassis_sim as cs

env = cs.make("ackermann", max_steps=300)
res = cs.evaluate(env, episodes=10, seed=0)
print(
    "n:",
    res.n,
    "success:",
    f"{res.success_rate:.0%}",
    "steps:",
    f"{res.mean_steps:.1f}",
    "collisions:",
    res.collisions,
)
print("distinct_maps:", res.distinct_maps, "maps_independent:", res.maps_independent)
print(res.markdown().splitlines()[2])

md = res.markdown(title="my run")
print("markdown title:", md.splitlines()[0], "| rows:", len(md.splitlines()))
```

```text
n: 10 success: 100% steps: 65.0 collisions: 0
distinct_maps: 1 maps_independent: False
| controller | ackermann |
markdown title: **my run** | rows: 11
```

`EvalResult` 暴露 `n`、`successes`、`collisions`、`timeouts`、`success_rate`、
`collision_rate`、`timeout_rate`、`mean_steps`、`mean_return`、
`min_obstacle_clearance`、`success_interval`、`success_ci`、`map_fingerprints`、
`distinct_maps`、`maps_independent`、`same_map_repeated`，外加 `to_dict()` 与
Markdown 渲染器 `markdown()`（即 `to_markdown()` 的别名）。`success_ci` 是
`max(success_rate - low, high - success_rate)`，即较大的那一侧 Wilson 半宽；区间本身
是非对称的。

- `distinct_maps` 是本轮里不同地图指纹的数量；`0` 表示地图无法指纹化。
- `maps_independent` 在 `n <= 1` 时为 `True`；否则要求每个回合都有非空指纹且互不相同。
- `same_map_repeated` 在出现重复地图时为 `True`。

传固定 reset options 字典是允许的，但那是同一张地图重放 n 次：评测器会往 stderr 打
警告，Markdown 表格把该行标成 `⚠固定地图`，置信区间也**不可**当作独立样本 CI 引用。
应当传可调用对象 —— `scenario_reset_options_factory(name, base_seed=0)` 按回合号现场
生成新地图，并把记账信息放进 `factory.stats`。`examples/eval_matrix.py` 就是可跑的
协议：3 控制器 × 3 底盘 × 10 张独立地图，两张 Markdown 表打到 stdout，JSON 落在 `/tmp`：

```bash
python examples/eval_matrix.py
```

`wilson_interval(successes, n, *, z=1.959963984540054)` 默认返回 95% 的 Wilson 得分
区间。

## 命令行（CLI）

包会装一个 `chassis-sim` 控制台脚本，`python -m chassis_sim` 是同一个程序：

```bash
chassis-sim list-scenarios
chassis-sim record --out demo.gif --scenario obstacles_sparse --controller pp --steps 300 --every 4 --fps 12 --seed 0
chassis-sim eval --scenario obstacles_sparse --controller dwa --episodes 10 --seed 0
chassis-sim render --out frame.png --scenario straight --seed 0

python -m chassis_sim list-scenarios
python -m chassis_sim eval --scenario dense_static --controller pp --episodes 5
```

| 子命令 | 参数 |
| --- | --- |
| `list-scenarios` | 无 |
| `record` | `--out demo.gif`、`--scenario S`、`--controller pp\|dwa`、`--steps N`、`--every K`、`--fps F`、`--seed S` |
| `eval` | `--scenario S`、`--controller pp\|dwa`、`--episodes N`、`--seed S` |
| `render` | `--out frame.png`、`--scenario S`、`--seed S` |

控制台脚本来自已安装发行版的 entry point，所以早于它创建的 editable 安装要重跑一次
`pip install -e ".[dev]"` 才会有 `chassis-sim` 命令；`python -m chassis_sim` 永远可用，
不依赖 entry point。

`record` 与 `render` 是需要 `viz` extra 的两个子命令。缺 `matplotlib` 或 `imageio`
时它们以**退出码 2** 结束，并打印一行中文提示（指向 `pip install "chassis-sim[viz]"`），
不是 traceback。

默认 `--scenario` 是 `corridor`，既不需要 `viz` 也不需要 `planning`，所以 `eval` 与
`record` 在最小安装下开箱可用。需要 `planning` extra 的场景（`obstacles_dense` 等）要
显式指定；缺 `numba` 时它们同样以**退出码 2** 结束，提示指向
`pip install "chassis-sim[planning]"`。核心场景的地图由各自 YAML 固定，因此对它跑
`eval` 会打印 `[WARN][eval] … 地图指纹全部相同` 并标 `固定地图`；要每回合重掷独立
地图，请用 `planning` 场景。

## 配置

`EnvConfig` 是带 44 个字段的 dataclass，也是所有旋钮的唯一事实源：底盘、动力学、动作
限值、观测、传感器、reward 分量、域随机化、世界边界。`cs.make(cfg, **kwargs)` 对每个
字段按 **kwargs > cfg 字段 > 库默认** 解析，`EnvConfig.to_yaml()` / `from_yaml()`
对整份配置做往返，因此一次实验靠一个快照文件就能复现。

```python
import chassis_sim as cs

cfg = cs.EnvConfig(chassis="ackermann", dt=0.1, v_max=1.2, max_steps=500)
print("fields:", len(cfg.__dataclass_fields__))
print("domain_rand_enabled:", cfg.domain_rand_enabled)

path = cfg.to_yaml("/tmp/run.yaml")
print("snapshot:", path.name, "->", path.read_text().splitlines()[0])
print("roundtrip:", cs.EnvConfig.from_yaml(path) == cfg)

env = cs.make(cfg, v_max=0.8)
print("kwargs > cfg:", env.unwrapped.chassis.v_max, "(cfg:", cfg.v_max, ")")
print("cfg carries max_steps:", env.max_steps)
print(
    "domain_rand: library default",
    cs.make("ackermann").unwrapped.domain_rand,
    "| cfg",
    cs.make(cs.EnvConfig(domain_rand_enabled=True)).unwrapped.domain_rand,
    "| kwargs",
    cs.make(
        cs.EnvConfig(domain_rand_enabled=True), domain_rand=False
    ).unwrapped.domain_rand,
)
```

```text
fields: 44
domain_rand_enabled: False
snapshot: run.yaml -> delta_max: 0.526
roundtrip: True
kwargs > cfg: 0.8 (cfg: 1.2 )
cfg carries max_steps: 500
domain_rand: library default False | cfg True | kwargs False
```

`EnvConfig.domain_rand_enabled` 默认是 **False**：不加噪才是库默认，要域随机化得自己
显式打开。解析分三层 —— `cs.make(domain_rand=True)` 压过 `cfg.domain_rand_enabled`，
后者压过库默认的 `False`。`to_yaml()` 写全 44 个字段，所以 `from_yaml()` 还原出
完全相同的配置，快照与来源对象比较相等。

## 从 0.2.x 迁移

0.2.x 不是可直接替换的升级：默认观测宽度、观测预设辅助、路径段名字、激光访问器、
回合数关键字、若干模块路径都变了，默认环境也不再自动装雷达。
[docs/migration-1.0.md](docs/migration-1.0.md) 是迁移指南 —— 一张 旧 → 新 对照表配
可跑的替换代码、观测预设的逐段移植，以及钉住那九个切片的 golden 测试。

## 场景

包内自带 11 个 YAML 场景。前五个是 core 场景，只要基础安装；后六个用 Hybrid-A* 与
Reeds-Shepp 规划，声明了 `requires: planning`。

| 场景 | 依赖 | Seed | 静态障碍 | 说明 |
| --- | --- | ---: | ---: | --- |
| `circle` | core | 0 | 0 | 圆形参考路径 |
| `corridor` | core | 0 | 2 | 窄走廊 |
| `dense_static` | core | 0 | 14 | 密集静态场 |
| `dynamic_crossing` | core | 0 | 0 | 动态障碍经 reset options 注入 |
| `straight` | core | 0 | 0 | 直线参考路径 |
| `obstacles_sparse` | `planning` | 0 | 0 | 生成的稀疏障碍场 |
| `obstacles_medium` | `planning` | 0 | 0 | 生成的中等障碍场 |
| `obstacles_dense` | `planning` | 0 | 0 | 生成的密集障碍场 |
| `obstacles_dense_reverse` | `planning` | 0 | 0 | 密集场 + 反向路径 |
| `foldback_reverse` | `planning` | 0 | 0 | 反向折返，过 cusp 过滤 |
| `maze_grid` | `planning` | 1 | 0 | 网格迷宫（YAML seed 1） |

`cs.make(scenario=...)` 会加载场景并翻译成 reset options，但它**不会**替你调
`reset()`：这些键落在 `env.default_reset_options` 里，要你自己交给 `reset()`。没有
任何隐式生效，场景名写错时 `ScenarioNotFoundError` 原样抛出，不静默忽略。

```python
import chassis_sim as cs

env = cs.make("ackermann", scenario="corridor", max_steps=100)
print("scenario:", env.scenario.name, "requires:", env.scenario.requires)
print("reset option keys:", sorted(env.default_reset_options))

obs, info = env.reset(seed=0, options=env.default_reset_options)
print(
    "obs:",
    obs.shape,
    "obstacles:",
    len(env.unwrapped._obstacles),
    "path:",
    env.unwrapped.bridge.get_path_global().shape,
)

for name in ("straight", "corridor", "dense_static"):
    scenario = cs.load_scenario(name)
    print(
        f"{name:14s} obstacles={len(scenario.obstacles):2d} "
        f"requires={scenario.requires} seed={scenario.seed}"
    )
```

```text
scenario: corridor requires: None
reset option keys: ['initial_heading', 'obstacles', 'path_waypoints']
obs: (128,) obstacles: 2 path: (141, 3)
straight       obstacles= 0 requires=None seed=0
corridor       obstacles= 2 requires=None seed=0
dense_static   obstacles=14 requires=None seed=0
```

同一个字典喂给每个回合就是同一张地图重放。做 demo 可以，做基准不行：
`scenario_reset_options_factory(name, base_seed=0)` 返回可调用对象，按回合号生成新
地图；`scenario_to_reset_options` 仍然是一次性形式。`cs.list_scenarios()` 与
`cs.load_scenario()` 都在顶层 API 里。

## 动力学

执行器模型有三档。用构造参数或 `EnvConfig.dynamics` 选，构造参数优先于配置。

| 档位 | 语义 | 默认值 |
| --- | --- | --- |
| `none` | 纯运动学；`dynamics.py` 既不 import 也不调用；`info["dynamics"]` 键不出现 | — |
| `limits` | 加速度与转向速率限值 | `accel_max = 1.5 m/s²`、`steer_rate_max = 3.0 rad/s` |
| `wheel` | 限值 + 一阶轮速滞后、死区、打滑 | `v_tau = 0.15 s`、`omega_tau = 0.10 s`、`v_deadband = 0.0`、`wheel_slip = 0.0` |

`EnvConfig.accel_max` 与 `EnvConfig.steer_rate_max` 默认是 `0.0`。对 `limits` 与
`wheel`，非正值表示“未设置”，解析成上表的 1.5 m/s² / 3.0 rad/s；`none` 不动配置值。
`wheel` 按 `v += (v_cmd - v) * (1 - exp(-dt / v_tau))` 更新，`ω` 同一套一阶更新，
然后乘 `(1 - wheel_slip)`；落在 `v_deadband` 里的指令被强制清零。

`none` 路径与开关之前的行为逐位一致：它不 import 动力学模块、不调用它、不消耗任何
随机数。下面这段把 `dynamics=None` 与 `dynamics="none"` 对拍 20 步并打印 `True`；它
通过 `obs_spec_describe()` 读里程计，而不是硬编码切片。

```python
import numpy as np

import chassis_sim as cs


def run(level, **overrides):
    cfg = cs.EnvConfig(chassis="diff_drive", dynamics=level, **overrides)
    env = cs.make(cfg, max_steps=10)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    return obs[env.obs_spec_describe()["odom"]], info.get("dynamics")


for level in ("none", "limits", "wheel"):
    odom, dynamics_info = run(level)
    print(f"{level:6s} odom={np.round(odom, 6)} info.dynamics={dynamics_info}")

default_env = cs.make("diff_drive", max_steps=20)
none_env = cs.make("diff_drive", dynamics="none", max_steps=20)
default_obs, _ = default_env.reset(seed=3)
none_obs, _ = none_env.reset(seed=3)
identical = np.array_equal(default_obs, none_obs)
for _ in range(20):
    action = np.array([0.4, 0.3], dtype=np.float32)
    default_obs, *_ = default_env.step(action)
    none_obs, *_ = none_env.step(action)
    identical = identical and np.array_equal(default_obs, none_obs)
print("dynamics=None vs 'none' obs bit-identical over 20 steps:", identical)
```

```text
none   odom=[1.2 0.  1.2 0. ] info.dynamics=None
limits odom=[0.15 0.   0.15 0.  ] info.dynamics={'level': 'limits', 'accel_max': 1.5, 'steer_rate_max': 3.0}
wheel  odom=[0.072987 0.       0.072987 0.      ] info.dynamics={'level': 'wheel', 'accel_max': 1.5, 'steer_rate_max': 3.0, 'v_tau': 0.15, 'omega_tau': 0.1, 'alpha_v': 0.486582880967408, 'alpha_omega': 0.6321205588285577, 'v_deadband': 0.0, 'wheel_slip': 0.0}
dynamics=None vs 'none' obs bit-identical over 20 steps: True
```

## 规划

`pip install "chassis-sim[planning]"` 会加上 `numba`。该 extra 提供 `MapGenerator`、
`hybrid_astar_search` 以及 Reeds-Shepp 原语（`reeds_shepp_path`、
`reeds_shepp_path_pose`、`reeds_shepp_length`）。

cusp 过滤器会拒掉倒车折返过短、或离前进段过近的参考路径。默认值来自
`config/curriculum.yaml`：

| 键 | 默认值 | 作用 |
| --- | ---: | --- |
| `min_cusp_fold_arc` | 0.8 m | 拒掉短于该值的极大倒车段 |
| `min_cusp_leg_separation` | 0.35 m | 拒掉离前进段近于该值的倒车段 |
| `cusp_min_arc_gap` | 1.0 m | 只比较弧长间隔不小于该值的段对 |

把阈值设成 `0` 即关闭该判据。被拒的候选会被重采样；`MapGenerator.cusp_rejections`
记录拒绝次数。

没有 numba 时 core 场景照常跑；planning 场景抛 `ImportError`，消息里带安装提示：

```python
import sys

import chassis_sim as cs
from chassis_sim.planning import MapGenerator, hybrid_astar_search, reeds_shepp_length
from chassis_sim.scenarios import scenario_reset_options_factory

print(
    "planning imports:",
    MapGenerator.__name__,
    hybrid_astar_search.__name__,
    reeds_shepp_length.__name__,
)

factory = scenario_reset_options_factory("obstacles_sparse", base_seed=0)
env = cs.make("ackermann", max_steps=100)
obs, info = env.reset(seed=0, options=factory(0))
print(
    "generated map:",
    f"obstacles={len(env.unwrapped._obstacles)}",
    f"path={env.unwrapped.bridge.get_path_global().shape}",
    f"factory.stats={factory.stats}",
)

sys.modules["numba"] = None
try:
    without_numba = scenario_reset_options_factory("obstacles_sparse", base_seed=0)
    without_numba(0)
except ImportError as exc:
    print("ImportError:", exc)
```

```text
planning imports: MapGenerator hybrid_astar_search reeds_shepp_length
generated map: obstacles=5 path=(152, 3) factory.stats={'attempts': 1, 'retries': 0, 'failures': 0, 'episodes': 1}
ImportError: 场景 'obstacles_sparse' 需要 Hybrid-A* 地图生成器（chassis_sim.planning），它依赖 numba。请先安装 planning extra：pip install chassis-sim[planning]
```

## 渲染

三层能力，三个命名空间：

| 层 | 入口 | 产物 |
| --- | --- | --- |
| 一个回合 | `cs.record(env, controller, out="demo.gif", every=4, fps=12)` | GIF（缺 `imageio` 时降级成帧目录） |
| 一张静图 | `cs.snapshot(env, "hero.png")` | 当前状态的 PNG |
| 画图原语 | `chassis_sim.render`：`render_frame`、`render_to_png`、`render_episode_frames`、`save_gif` | PNG / GIF，且能控制整幅画面 |

`chassis_sim.render` 用 matplotlib 的 `Agg` 后端画当前桥状态，因此可无头运行。
`render_to_png()` 写一帧并返回路径；`render_episode_frames()` 让桥走完一段动作序列
并写帧（恒含最后一步）；`save_gif()` 把帧拼成 GIF。matplotlib 是惰性 import，只在渲染
时需要。`imageio` 可选：缺了就由 `save_gif()` 写成帧目录并打一条警告，不抛异常。
`examples/gallery.py` 建在同一批原语上，产出图库那一节的素材。

```python
from pathlib import Path

import numpy as np

import chassis_sim as cs
from chassis_sim.render import render_episode_frames, render_to_png

env = cs.make("ackermann", max_steps=50)
obs, info = env.reset(seed=0)
out_dir = Path("/tmp/chassis_sim_readme_render")
out_dir.mkdir(parents=True, exist_ok=True)

png = render_to_png(env.unwrapped.bridge, out_dir / "frame.png", dpi=80)
print("png:", png.name, "exists:", png.exists())

actions = [
    np.array([0.2 * np.sin(index / 5.0), 0.4], dtype=np.float64) for index in range(10)
]
frames = render_episode_frames(
    env.unwrapped.bridge, actions, out_dir, every=5, prefix="step", dpi=80
)
print("frames:", [frame.name for frame in frames])
print("all non-empty:", all(frame.stat().st_size > 0 for frame in frames))
```

```text
png: frame.png exists: True
frames: ['step_00005.png', 'step_00010.png']
all non-empty: True
```

## 示例

| 脚本 | 做什么 |
| --- | --- |
| `examples/dynamic_demo.py` | 挂上雷达后对比 `dynamic_crossing` 的静态 / 动态；校验剔除动态障碍后重放逐位一致 |
| `examples/eval_matrix.py` | 3 控制器 × 3 底盘 × 10 张独立地图（`max_steps=300`）；输出 Markdown 与 JSON |
| `examples/gallery.py` | 重建图库的十一份素材（5 段 GIF @ 场景 seed 0，6 张场景静图） |
| `examples/render_demo.py` | 60 步、60 张 PNG 帧、hero 静图与渲染耗时 |
| `examples/scenario_demo.py` | `straight` / `corridor` / `dense_static` 各跑 50 步随机动作，挂载雷达 |
| `examples/sensors_demo.py` | IMU、轮速计与带延迟/丢帧的参数化 LiDAR |

在仓库根目录用装了本包的解释器运行。

## 已知限制

1. **默认环境不装雷达，`d_min` 不是测量值。** `cs.make("ackermann")` 不装配任何
   传感器，因此 `info["d_min"]` 恒等于 `bridge.max_range`（默认 10.0 m），reward 的
   obstacle 项退化为常数。要拿到真实净空与非恒定障碍惩罚，必须显式声明
   `sensors=[cs.Lidar(...)]`。
2. **没有底盘就构造不了控制器。** `PurePursuitController()` 与 `DwaController()` 在
   缺 `chassis=` / `kind=` 时抛 `TypeError`，且本包不给隐式阿克曼默认：麦轮环境配上
   阿克曼标定的控制器会把每条指令静默缩放。用 `cs.make_controller("pp", env=env)`
   （或 `chassis=env.unwrapped.chassis`），或者让 `cs.evaluate(env, episodes=...)`
   把纯追踪绑到环境自己的底盘上。
3. **静图入口叫 `cs.snapshot()`，不叫 `cs.render()`。** `chassis_sim.render` 是画图
   子包，而 import 子模块会把父包同名属性重绑成 module 对象 —— 任何装成
   `chassis_sim.render` 的函数，都会在渲染器第一次被 import 时被覆盖（`cs.record()`
   就会触发这次 import）。因此另取名字；画图原语仍在 `chassis_sim.render` 里。
4. **`ChassisSimEnv(device=...)` 是无效形参。** 编码器早在 1.0.0 之前就移进了 Actor，
   观测构造是纯 NumPy，所以 `device` 不再选择任何东西；保留它只是为了让仍然透传它的
   训练脚本还能 import。别指望它能搬动计算。
5. **离散碰撞检测，没有扫掠体。** 碰撞是逐步对当前车体 OBB 与圆/有向矩形做离散判定
   （带安全余量）；没有扫掠体 / 连续（CCD）检测，所以高速倒车仍可能在两帧之间蹭到角。
6. **观测里的缺帧是零阶保持，不是真值。** 帧被延迟或丢弃时 `LidarSegment` 重复上一
   帧有效扫描（第一帧之前回落到 `max_range`）。观测本身不带有效位，所以不读 `info`
   里的 `scan_meta`，下游分不清是新扫描还是保持值。
7. **DWA 的挡位推断默认关闭。** `gear_mode="auto"` 会把候选截到单一方向，并对
   `heading_lookahead` 敏感；配上上面那条离散碰撞检测，硬选挡位可能在一次本可由全域
   候选避开的机动里撞上。
8. **没有相机。** 没有 RGB、深度或分割传感器。
9. **core 场景一张地图一个。** core 场景是固定 YAML 地图；只有 planning 场景（以及自
   定义的 callable）才逐回合生成新地图。
10. **图库 GIF 是 10 Hz 环路的 12 fps 抽样。** 片段跳过了帧与帧之间的 tick，播放快于
    实时，所以它是行为示意，不是控制序列的记录；要按当前代码重建请重跑
    `examples/gallery.py`。

## 许可证、贡献与引用

MIT，见 `LICENSE`。

测试与 lint：

```bash
env -u PYTHONPATH pytest -q
ruff check .
```

CI 在 Python 3.10、3.11、3.12 上跑测试套件（`.github/workflows/ci.yml`）。

如果本包对学术工作有帮助，请引用本仓库：

```bibtex
@software{chassis_sim_2026,
  author = {zhao6663},
  title = {chassis-sim: Gymnasium environments for Ackermann, differential-drive
           and mecanum chassis},
  year = {2026},
  url = {https://github.com/zhao6663/chassis-sim}
}
```

---

标题、安装段与所有版本口径都指向 1.0.0。本文每个 `python` 代码块都按原样执行过，
其下的 `text` 块就是那次运行的 stdout。
