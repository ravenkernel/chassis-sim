# chassis-sim

[English](https://github.com/zhao6663/chassis-sim/blob/main/README.md) | 简体中文

[![CI](https://github.com/zhao6663/chassis-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/zhao6663/chassis-sim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/zhao6663/chassis-sim)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**一个面向三类底盘的、体量小且依赖轻的 Gymnasium 基准** —— Ackermann 转向、差速驱动
（differential drive）与麦克纳姆轮（mecanum） —— 提供归一化动作、纯 NumPy 仿真器、
可组合的观测规约、声明式传感器，以及用于可复现评测的参考控制器。

版本 0.2.1。Python 3.10+。

## 概览

每个环境都暴露同一个归一化动作空间 `Box(-1, 1, (d,), float32)`。由底盘把这个归一化
动作换算成物理量；环境从不要求用户自己给电压、轮上力矩或原始转向角。

| Gymnasium id | 底盘 | 动作维度 | 动作布局 | 默认观测维度 | 默认配置下的物理限制 |
| --- | --- | ---: | --- | ---: | --- |
| `ChassisSim-Ackermann-v0` | Ackermann | 2 | `[steer, speed]` | 128 | `v ∈ [-0.6, 1.2] m/s`, `δ ∈ [-0.526, 0.526] rad`, `|ω| ≤ 4.805 rad/s` |
| `ChassisSim-DiffDrive-v0` | 差速驱动 | 2 | `[speed, yaw]` | 128 | `v ∈ [-1.2, 1.2] m/s`, `|ω| ≤ 1.0 rad/s` |
| `ChassisSim-Mecanum-v0` | 麦轮 | 3 | `[vx, vy, yaw]` | 129 | `v_x ∈ [-1.2, 1.2] m/s`, `|vy| ≤ 0.5 m/s`, `|ω| ≤ 1.0 rad/s` |

上表的限制就是默认 `EnvConfig` 下 `env.chassis.describe()["limits"]` 报出的值，环境
内部并不另存一份。换一个 `EnvConfig`（或换一个 `Chassis` 实例）会同时改变物理映射与
动作语义。对 Ackermann 底盘，`delta_max` 与 `curvature_action` 决定第一个动作轴表示
的是转向角还是曲率。

默认观测是四段式规约 `ObsSpec.default(action_dim, path_points=30)` —— 里程计、参考
路径、目标，以及上一步的归一化动作 —— 因此在两台 2 维动作底盘上是 128 维，在麦轮上是
129 维。0.1.x 系列那个 494 维向量不再是默认值；它通过 `ObsLayout.as_spec()` 作为显式
选择保留，详见[观测与动作契约](#观测与动作契约)。

## 安装

从 PyPI 安装：

```bash
pip install "chassis-sim[viz]"
```

去掉 `[viz]` 后缀即为核心安装；需要 Hybrid-A* 地图生成器则用 `[planning]`。要固定到
某个版本的源码包？每个 tag 的 [GitHub release](https://github.com/zhao6663/chassis-sim/releases)
页面都挂着对应的 wheel。0.1.x 系列（tag `v0.1.4`）属于迁移场景：它仍在用旧的
494 维观测和旧的单配置（single-config）传感器 API。

在 checkout 里连开发工具链一起装：

```bash
git clone https://github.com/zhao6663/chassis-sim.git
cd chassis-sim
pip install -e ".[dev]"
```

### 可选 extras

| Extra | 安装内容 | 用途 |
| --- | --- | --- |
| `planning` | `numba>=0.56` | Hybrid-A* 地图生成，供规划类场景使用 |
| `viz` | `matplotlib>=3.7`, `rich>=13.0`, `imageio>=2.28` | 渲染、GIF 导出，以及 rich 终端辅助 |
| `dev` | `pytest>=7.0`, `ruff>=0.4` | 测试套件与 linter |

核心安装只依赖 `numpy`、`gymnasium` 与 `pyyaml`。`matplotlib` 只在渲染时需要，
`imageio` 只在导出 GIF 时需要；两者都在 `viz` extra 里，也就是 `examples/gallery.py`
所需的那一组。

0.2.x 系列只内置一个仿真器：纯 NumPy 的 `SimpleBridge`。不需要任何外部仿真器，也没有
`irsim` extra：ir-sim 与 ROS 2 后端不属于本包（它们仍留在上游训练仓库中）。
`PhysicalCommandBridge` 就是为这类后端预留、并有文档说明的接缝（seam）。

## 快速开始

```python
import gymnasium as gym
import numpy as np

import chassis_sim  # noqa: F401  (registers the three environment ids)

env = gym.make("ChassisSim-Ackermann-v0", max_steps=100)
obs, info = env.reset(seed=0)
print("obs:", obs.shape, obs.dtype)
print("action_space:", env.action_space)
print("obs_spec_describe:", env.unwrapped.obs_spec_describe())

total_reward = 0.0
steps = 0
terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.3], dtype=np.float32)
    )
    total_reward += float(reward)
    steps += 1

print("steps=" + str(steps) + " total_reward=" + format(total_reward, ".3f"))
print("terminated=" + str(terminated) + " truncated=" + str(truncated))
print("termination_reason:", info["termination_reason"])
print("info:", {key: info[key] for key in ("step", "collision", "goal_reached", "d_min")})
env.close()
```

运行后输出：

```text
obs: (128,) float32
action_space: Box(-1.0, 1.0, (2,), float32)
obs_spec_describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 128, None)}
steps=100 total_reward=-28.756
terminated=False truncated=True
termination_reason: timeout
info: {'step': 100, 'collision': False, 'goal_reached': False, 'd_min': 4.445171356201172}
```

注册的三个 id 是 `ChassisSim-Ackermann-v0`、`ChassisSim-DiffDrive-v0` 与
`ChassisSim-Mecanum-v0`。它们共用同一套观测规约机制、同样的奖励与终止逻辑；不同的
只有底盘、动作宽度，以及 `act` 段的宽度。

`ChassisSimEnv` 在自己的 `max_steps` 上终止；`gym.make` 不会额外套一层 `TimeLimit`
（注册时 `max_episode_steps=None`）。想缩短回合就传 `max_steps=...`。

## 图库

十一个素材，全部由一条命令在场景 seed 0 下重新生成：

```bash
python examples/gallery.py --out docs/media
```

五段动画是真实闭环运行的 12 fps 回放：控制器从 seed 0 的 reset 状态驱动环境，再把
录下来的动作序列从完全相同的 reset 状态重放一遍，所以每段动画都可复现，且每段都到达
目标。六张静态图是某个场景的初始状态（`reset` 之后的第 0 步，底盘 `ackermann`，无控制
器），画出障碍场、参考路径、车体，以及角落里的底盘名与场景名。每张图都链接到
`docs/media/` 下的原尺寸文件。

### 底盘运动学 —— 一条参考路径，一个控制器

三台底盘都用同一个 `pure_pursuit` 控制器跑同一条 `obstacles_sparse` 路径，所以唯一
变化的只有运动学；三者都在 76 步到达目标。

<table>
<tbody>
<tr>
<td align="center" width="33%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_ackermann.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_ackermann.gif" width="250" alt="Ackermann 底盘跟踪 obstacles_sparse 路径"></a><br><code>chassis_ackermann.gif</code><br><em>obstacles_sparse · ackermann · pure_pursuit · 76 步到达目标</em></td>
<td align="center" width="33%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_diff_drive.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_diff_drive.gif" width="250" alt="差速驱动底盘跟踪同一条 obstacles_sparse 路径"></a><br><code>chassis_diff_drive.gif</code><br><em>obstacles_sparse · diff_drive · pure_pursuit · 76 步到达目标</em></td>
<td align="center" width="33%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_mecanum.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/chassis_mecanum.gif" width="250" alt="麦轮底盘跟踪同一条 obstacles_sparse 路径"></a><br><code>chassis_mecanum.gif</code><br><em>obstacles_sparse · mecanum · pure_pursuit · 76 步到达目标</em></td>
</tr>
</tbody>
</table>

前轮转向受最小转弯半径约束，因此 Ackermann 车过弯时必须外扩走线。差速驱动没有横向
自由度，转向完全来自左右轮速差，在同一条路径上看起来是更紧的折线。麦轮多了一个正交
速度分量，能直接在车体上做横向修正，所以它贴合参考线最紧。

### 控制器 —— 一台底盘，一个场景

两段动画都是 `ackermann` 底盘跑 `dense_static`，差别只在控制器；两者都到达目标，DWA
用的步数更少。

<table>
<tbody>
<tr>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/controller_pure_pursuit.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/controller_pure_pursuit.gif" width="400" alt="纯跟踪控制器在 dense_static 路径上"></a><br><code>controller_pure_pursuit.gif</code><br><em>dense_static · ackermann · pure_pursuit · 104 步到达目标</em></td>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/controller_dwa.gif"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/controller_dwa.gif" width="400" alt="DWA 控制器在 dense_static 路径上"></a><br><code>controller_dwa.gif</code><br><em>dense_static · ackermann · dwa · 91 步到达目标</em></td>
</tr>
</tbody>
</table>

纯跟踪（pure pursuit）用一段前视圆弧换来平滑的跟踪，但会切角，车体经过时离障碍更近。
DWA 用速度窗口采样加上障碍代价，走更保守的线，用 91 步而不是 104 步完成。

### 场景 —— 俯视图

<table>
<tbody>
<tr>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_straight.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_straight.png" width="400" alt="空旷的 straight 场景，画出参考路径与底盘轮廓"></a><br><code>scene_straight.png</code><br><em>straight · seed 0 · reset 后第 0 步</em></td>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_corridor.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_corridor.png" width="400" alt="corridor 场景：两道长墙把可行带压窄"></a><br><code>scene_corridor.png</code><br><em>corridor · seed 0 · reset 后第 0 步</em></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_dense_static.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_dense_static.png" width="400" alt="dense_static 场景：14 个障碍紧贴参考路径"></a><br><code>scene_dense_static.png</code><br><em>dense_static · seed 0 · reset 后第 0 步</em></td>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_obstacles_dense.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_obstacles_dense.png" width="400" alt="Hybrid-A* 地图生成器产出的 obstacles_dense 场景"></a><br><code>scene_obstacles_dense.png</code><br><em>obstacles_dense · seed 0 · reset 后第 0 步</em></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_dynamic_crossing.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_dynamic_crossing.png" width="400" alt="dynamic_crossing 场景，运动圆处在其 t=0 位姿"></a><br><code>scene_dynamic_crossing.png</code><br><em>dynamic_crossing · seed 0 · reset 后第 0 步</em></td>
<td align="center" width="50%"><a href="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_obstacles_sparse.png"><img src="https://raw.githubusercontent.com/zhao6663/chassis-sim/main/docs/media/scene_obstacles_sparse.png" width="400" alt="Hybrid-A* 地图生成器产出的 obstacles_sparse 场景"></a><br><code>scene_obstacles_sparse.png</code><br><em>obstacles_sparse · seed 0 · reset 后第 0 步</em></td>
</tr>
</tbody>
</table>

`straight` 是干净基线：没有障碍，y = 5 处一条 4 m 的参考线。`corridor` 把可行区域压进
两个长矩形之间 0.6 m 宽的带子里，离墙余量因此能直接读出来。`dense_static` 在参考路径
旁紧贴 14 个圆与矩形，全部手写，不需要规划器。`obstacles_dense` 与 `obstacles_sparse`
来自 Hybrid-A* 地图生成器，其障碍场按 seed 现场生成而不是存在 YAML 里：dense 在路径
附近抽 9–15 个障碍，sparse 在开阔空间里抽 2–5 个。`dynamic_crossing` 有一个半径 0.25 m
的圆以 0.3 m/s 横穿路径，20 s 一个乒乓周期，图中画的是它 t = 0 的位姿。

## 观测与动作契约

观测是一个 `float32` 向量，由一份有序的段列表拼装而成。默认列表就是
`ObsSpec.default(action_dim, path_points=30)`：

`ObsSpec((Odom(), Path(points=path_points), Goal(), Act(action_dim=action_dim)))`
—— 依次是里程计、参考路径、目标，以及上一步的归一化动作。

| 段 | 构造函数 | 宽度 | 含义 |
| --- | --- | ---: | --- |
| `Odom` | `Odom()` | 4 | 车体系下的 `[dx*10, dy*10, v, ω]` |
| `Path` | `Path(points=30)` | `points * 4` | 30 个点 × `[rho, sin_phi, cos_phi, dtheta]` |
| `Goal` | `Goal()` | 2 | 目标点在车体系下的位置 |
| `Act` | `Act(action_dim)` | `action_dim` | 上一步的归一化动作 |
| `Err` | `Err()` | 2 | 归一化横向误差 + 航向误差 |
| `GoalSig` | `GoalSig()` | 2 | 目标航向误差的 `sin` / `cos` |
| `PathValid` | `PathValid()` | 1 | 真实路径点所占比例，`[0, 1]` |
| `PathProgress` | `PathProgress()` | 1 | 弧长完成度 `[0, 1]`，单调不减 |
| `Lidar` | `Lidar(name, sensors, pool="none", ray_counts=None)` | Σ 射线数，或合并后 | 具名传感器的 LiDAR 距离，经同一套规约读取 |
| `Custom` | `Custom(name, dim, fn)` | `dim` | 你的可调用对象，入参为 `ObsContext` |

`spec.resolved(sensor_meta=None)` 返回一个 `ResolvedObsSpec`，带 `total_dim`、
`describe()`（名称 → `slice`）与 `build(ctx)`。环境用 `env.obs_spec_describe()` 暴露同一
份映射，对应它正在使用的规约。段名必须唯一；`Custom` 段的可调用对象若返回的宽度不对，
会抛 `ValueError`，并指明段名、声明的宽度与实际宽度。

默认规约在两台 2 维动作底盘上是 128 维、在麦轮上是 129 维，因为只有 `act` 段变宽：

| 段 | `d=2` 的切片 | `d=3` 的切片 |
| --- | --- | --- |
| `odom` | `[0:4]` | `[0:4]` |
| `path` | `[4:124]` | `[4:124]` |
| `goal` | `[124:126]` | `[124:126]` |
| `act` | `[126:128]` | `[126:129]` |

```python
import gymnasium as gym
import numpy as np

import chassis_sim  # noqa: F401
from chassis_sim.envs import ObsLayout
from chassis_sim.obs import Act, Goal, ObsSpec, Odom, Path

for env_id in ("ChassisSim-Ackermann-v0", "ChassisSim-DiffDrive-v0", "ChassisSim-Mecanum-v0"):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    print(f"{env_id:26s} obs={obs.shape} {obs.dtype} action_dim={env.action_space.shape[0]}")
    print("  describe:", env.unwrapped.obs_spec_describe())
    env.close()

for action_dim in (2, 3):
    spec = ObsSpec.default(action_dim=action_dim, path_points=30)
    resolved = spec.resolved()
    print("default spec action_dim=" + str(action_dim), "total=" + str(resolved.total_dim),
          "segments=" + str([segment.name for segment in spec.segments]))

legacy = ObsLayout(action_dim=2, max_path_points=30).as_spec()
print("explicit 494 preset:", legacy.resolved().total_dim, legacy.resolved().describe())

custom = ObsSpec((Odom(), Path(points=4), Goal(), Act(action_dim=2)))
print("path_points=4 total:", custom.resolved().total_dim)
```

输出：

```text
ChassisSim-Ackermann-v0    obs=(128,) float32 action_dim=2
  describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 128, None)}
ChassisSim-DiffDrive-v0    obs=(128,) float32 action_dim=2
  describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 128, None)}
ChassisSim-Mecanum-v0      obs=(129,) float32 action_dim=3
  describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 129, None)}
default spec action_dim=2 total=128 segments=['odom', 'path', 'goal', 'act']
default spec action_dim=3 total=129 segments=['odom', 'path', 'goal', 'act']
explicit 494 preset: 494 {'lidar': slice(0, 360, None), 'odom': slice(360, 364, None), 'path': slice(364, 484, None), 'goal': slice(484, 486, None), 'act': slice(486, 488, None), 'err': slice(488, 490, None), 'goal_sig': slice(490, 492, None), 'path_valid': slice(492, 493, None), 'path_progress': slice(493, 494, None)}
path_points=4 total: 24
```

不要手工给向量切下标。`env.obs_spec_describe()` 才是契约：

```python
import numpy as np

from chassis_sim.envs import ChassisSimEnv
from chassis_sim.obs import Act, Custom, Goal, ObsSpec, Odom, Path


def bias(ctx):
    return np.array([np.sum(np.abs(ctx.odom_rel)) + 0.25])


spec = ObsSpec((
    Odom(),
    Path(points=30),
    Goal(),
    Act(action_dim=2),
    Custom("bias", 1, bias),
))
env = ChassisSimEnv("ackermann", max_steps=20, obs_spec=spec)
obs, info = env.reset(seed=0)
print("obs:", obs.shape, "describe:", env.obs_spec_describe())
for _ in range(3):
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.3], dtype=np.float32)
    )
print("bias value:", float(obs[env.obs_spec_describe()["bias"]][0]))

wrong = ObsSpec((Odom(), Custom("bias", 2, bias)))
try:
    ChassisSimEnv("ackermann", max_steps=5, obs_spec=wrong).reset(seed=0)
except ValueError as exc:
    print("ValueError:", exc)
```

输出：

```text
obs: (129,) describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 128, None), 'bias': slice(128, 129, None)}
bias value: 0.970000147819519
ValueError: Custom('bias').build: 期望宽度 2，实际 1
```

### 显式的 494 维预设

按 0.1.x 布局写的代码只要显式索取就能继续用。
`ObsLayout(action_dim, max_path_points).as_spec()` 返回九段式规约（lidar 360 + odom 4 +
path 120 + goal 2 + act + err 2 + goal_sig 2 + path_valid 1 + path_progress 1），在
`action_dim=2` 时共 494 维：

```python
import gymnasium as gym

import chassis_sim
from chassis_sim.envs import ObsLayout

env = gym.make("ChassisSim-Ackermann-v0", obs_spec=ObsLayout(2, 30).as_spec()).unwrapped
obs, info = env.reset(seed=0)
print("obs:", obs.shape, obs.dtype)
print("describe:", env.obs_spec_describe())
env.close()
```

输出：

```text
obs: (494,) float32
describe: {'lidar': slice(0, 360, None), 'odom': slice(360, 364, None), 'path': slice(364, 484, None), 'goal': slice(484, 486, None), 'act': slice(486, 488, None), 'err': slice(488, 490, None), 'goal_sig': slice(490, 492, None), 'path_valid': slice(492, 493, None), 'path_progress': slice(493, 494, None)}
```

动作永远是 `Box(-1, 1, (d,), float32)`。归一化值在底盘内部变成物理指令：
`Chassis.command()` 把动作映射成 `(v, ω, vy)`，`Chassis.step()` 推进运动学状态。对
Ackermann 底盘，`step()` 用上一拍的转向角计算偏航角速度、然后再存下新值，所以按构造就
存在一拍（one-tick）的转向滞后。

### 物理指令桥接

第三方仿真器（ir-sim、ROS/Gazebo、主仓库的 `NumpySimBridge`）接受的是**物理**指令，
而不是归一化动作。`PhysicalCommandBridge` 用 `chassis.physical_action()` 做映射（绝不
用 `step()` —— 转向滞后归后端自己管），传感器则原样转发。向量顺序由底盘自己决定：

| 底盘 | 物理向量 |
| --- | --- |
| `ackermann` | `[delta (rad), v (m/s)]` |
| `diff_drive` | `[v (m/s), omega (rad/s)]` |
| `mecanum` | `[v (m/s), vy (m/s), omega (rad/s)]` |

`bridge.step()` 接受的仍然是**归一化**动作，把 `chassis.physical_action(action)` 交给
`inner.step()`；`bridge.to_physical(action)` 把这份映射作为纯函数暴露出来，所以同一个
归一化动作无论执行器状态如何都产出同一个物理向量。执行器滞后必须**恰好施加一次**，由
内层物理仿真器施加：适配器从不推进底盘状态，也从不调用 `chassis.reset_state()`。

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
bridge.step([0.3, 0.6])  # normalized action in, physical vector out
print("inner received:", inner.commands)
```

输出：

```text
physical_action([1.0, 0.5]): [0.52558379 0.5       ]
inner received: [[0.1722752169535286, 0.6]]
```

## 场景

包内附带十一个 YAML 场景。前五个是核心场景，只需要最小安装。后六个用到 Hybrid-A* 与
Reeds-Shepp 规划，声明了 `requires: planning`。

| 场景 | 依赖 | Seed | 静态障碍数 | 说明 |
| --- | --- | ---: | ---: | --- |
| `circle` | core | 0 | 0 | 圆形参考路径 |
| `corridor` | core | 0 | 2 | 狭窄走廊 |
| `dense_static` | core | 0 | 14 | 密集静态障碍场 |
| `dynamic_crossing` | core | 0 | 0 | 动态障碍通过 reset options 传入 |
| `straight` | core | 0 | 0 | 直线参考路径 |
| `obstacles_sparse` | `planning` | 0 | 0 | 生成的稀疏障碍场 |
| `obstacles_medium` | `planning` | 0 | 0 | 生成的中等密度障碍场 |
| `obstacles_dense` | `planning` | 0 | 0 | 生成的密集障碍场 |
| `obstacles_dense_reverse` | `planning` | 0 | 0 | 密集障碍场，路径取向含倒车 |
| `foldback_reverse` | `planning` | 0 | 0 | 倒车折返，经 cusp 过滤 |
| `maze_grid` | `planning` | 1 | 0 | 网格迷宫（YAML seed 1） |

可以把场景只加载一次，也可以构造按回合生成 reset options 的工厂：

```python
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.scenarios import (
    load_scenario,
    scenario_reset_options_factory,
    scenario_to_reset_options,
)

names = (
    "circle", "corridor", "dense_static", "dynamic_crossing", "straight",
    "obstacles_sparse", "obstacles_medium", "obstacles_dense",
    "obstacles_dense_reverse", "foldback_reverse", "maze_grid",
)
for name in names:
    scenario = load_scenario(name)
    print(f"{name:24s} requires={str(scenario.requires):8s} "
          f"obstacles={len(scenario.obstacles):2d} seed={scenario.seed}")

scenario = load_scenario("corridor")
options = scenario_to_reset_options(scenario)
env = ChassisSimEnv("ackermann", max_steps=100)
obs, info = env.reset(seed=0, options=options)
print("corridor reset:", obs.shape, "obstacles=", len(env._obstacles),
      "path=", env.bridge.get_path_global().shape)

factory = scenario_reset_options_factory("obstacles_sparse", base_seed=0)
env = ChassisSimEnv("ackermann", max_steps=100)
for episode in range(2):
    obs, info = env.reset(seed=episode, options=factory(episode))
    print(f"episode={episode} obstacles={len(env._obstacles)} "
          f"path={env.bridge.get_path_global().shape}")
print("factory.stats:", factory.stats)
```

输出：

```text
circle                   requires=None     obstacles= 0 seed=0
corridor                 requires=None     obstacles= 2 seed=0
dense_static             requires=None     obstacles=14 seed=0
dynamic_crossing         requires=None     obstacles= 0 seed=0
straight                 requires=None     obstacles= 0 seed=0
obstacles_sparse         requires=planning obstacles= 0 seed=0
obstacles_medium         requires=planning obstacles= 0 seed=0
obstacles_dense          requires=planning obstacles= 0 seed=0
obstacles_dense_reverse  requires=planning obstacles= 0 seed=0
foldback_reverse         requires=planning obstacles= 0 seed=0
maze_grid                requires=planning obstacles= 0 seed=1
corridor reset: (128,) obstacles= 2 path= (141, 3)
episode=0 obstacles=5 path=(152, 3)
episode=1 obstacles=4 path=(168, 3)
factory.stats: {'attempts': 2, 'retries': 0, 'failures': 0, 'episodes': 2}
```

`scenario_to_reset_options(scenario)` 返回一个普通 dict。每个回合都复用这个 dict，等于
重放同一张地图；下面评测一节会解释为什么这很重要。
`scenario_reset_options_factory(name, base_seed=0)` 返回一个可调用对象，按回合索引生成
新地图，并把记账信息暴露在 `factory.stats` 里。

## 评测

`evaluate()` 让一个控制器跑 `n_episodes` 个回合，返回 `EvalResult`。
`wilson_interval(successes, n)` 默认返回 95% 的 Wilson 得分区间
（`z = 1.959963984540054`）。

```text
evaluate(env_factory, controller, *, n_episodes, seed=0, max_steps=None,
         reset_options=None, labels=None, allow_mismatch=False) -> EvalResult
wilson_interval(successes, n, *, z=1.959963984540054) -> (low, high)
```

`EvalResult` 暴露 `n`、`successes`、`collisions`、`timeouts`、`success_rate`、
`collision_rate`、`timeout_rate`、`mean_steps`、`mean_return`、
`min_obstacle_clearance`、`success_interval`、`success_ci`、`map_fingerprints`、
`distinct_maps`、`maps_independent` 与 `same_map_repeated`。`success_ci` 取
`max(success_rate - low, high - success_rate)`，即较大的那个 Wilson 半宽；区间本身是
不对称的。

- `distinct_maps` 是本次运行中不同地图指纹的个数。`0` 表示地图没能生成指纹。
- 当 `n <= 1`，或每个回合都有非空指纹且所有指纹两两不同时，`maps_independent` 为
  `True`。
- 运行中出现重复地图时，`same_map_repeated` 为 `True`。

固定的 reset-options dict 是允许的，但它会重放同一张地图。评测器会往 stderr 打警告，
此时的置信区间不能当成独立样本 CI 使用。用固定 dict 跑两个回合会打印：

```python
import contextlib
import io

from chassis_sim.controllers import PurePursuitController
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.eval import evaluate
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options

scenario = load_scenario("obstacles_sparse")
fixed = scenario_to_reset_options(scenario, seed=scenario.seed)

env = ChassisSimEnv("ackermann", max_steps=500)
controller = PurePursuitController(chassis=env.chassis)

buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    result = evaluate(
        env, controller, n_episodes=2, seed=0, reset_options=fixed
    )

print("distinct_maps:", result.distinct_maps)
print("maps_independent:", result.maps_independent)
print("same_map_repeated:", result.same_map_repeated)
for line in buf.getvalue().splitlines():
    if line.startswith("[WARN][eval]"):
        print(line)
```

输出：

```text
distinct_maps: 1
maps_independent: False
same_map_repeated: True
[WARN][eval] reset_options 是固定 dict：2 个回合跑的是**同一张地图**（回合 seed 只推进 env 内部 RNG，不重掷地图）⇒ success_ci 只是该图上的重放方差，不代表独立样本，不可用于控制器对比。要每回合一张独立地图请传 callable(episode_index)（例：scenario_reset_options_factory('obstacles_dense', base_seed=0)）。
```

改成传可调用对象，如下例。这就是“十张独立地图上的 CI”与“同一张地图重放十次的 CI”
之间的区别。

```python
from chassis_sim.controllers import PurePursuitController
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.eval import evaluate, wilson_interval
from chassis_sim.scenarios import scenario_reset_options_factory

probe = ChassisSimEnv("ackermann", max_steps=500)
controller = PurePursuitController(chassis=probe.chassis)
factory = scenario_reset_options_factory("obstacles_sparse", base_seed=0)
result = evaluate(
    lambda: ChassisSimEnv("ackermann", max_steps=500),
    controller,
    n_episodes=10,
    seed=0,
    max_steps=500,
    reset_options=factory,
)
print(f"success={result.success_rate:.1%} "
      f"interval=({result.success_interval[0]:.1%}, {result.success_interval[1]:.1%}) "
      f"steps={result.mean_steps:.1f} collisions={result.collision_rate:.1%}")
print(f"distinct_maps={result.distinct_maps} maps_independent={result.maps_independent} "
      f"same_map_repeated={result.same_map_repeated}")
for successes in (10, 9, 8, 0):
    low, high = wilson_interval(successes, 10)
    print(f"wilson {successes}/10 -> [{low:.4f}, {high:.4f}]")
```

输出：

```text
success=100.0% interval=(72.2%, 100.0%) steps=94.3 collisions=0.0%
distinct_maps=10 maps_independent=True same_map_repeated=False
wilson 10/10 -> [0.7225, 1.0000]
wilson 9/10 -> [0.5958, 0.9821]
wilson 8/10 -> [0.4902, 0.9433]
wilson 0/10 -> [0.0000, 0.2775]
```

### 参考矩阵

下表就是用这套协议跑出来的：`n=10` 张独立地图、`max_steps=500`、`base_seed=0`、
Wilson 95% CI。成功率写成 `rate ± success_ci`（较大的那个 Wilson 半宽）；平均步数与
碰撞率覆盖全部十个回合。每一格都满足 `distinct_maps=10` 与 `maps_independent=True`。
随机基线在归一化动作空间里均匀采样。

| 场景 | 底盘 | 控制器 | 成功率（Wilson 95%） | 平均步数 | 碰撞率 |
| --- | --- | --- | ---: | ---: | ---: |
| `obstacles_sparse` | Ackermann | 纯跟踪（pure pursuit） | 100% ± 27.8 | 94.3 | 0% |
| `obstacles_sparse` | Ackermann | DWA | 100% ± 27.8 | 76.9 | 0% |
| `obstacles_sparse` | Ackermann | 随机 | 0% ± 27.8 | 303.3 | 70% |
| `obstacles_sparse` | 差速驱动 | 纯跟踪（pure pursuit） | 100% ± 27.8 | 93.1 | 0% |
| `obstacles_sparse` | 差速驱动 | DWA | 100% ± 27.8 | 76.0 | 0% |
| `obstacles_sparse` | 差速驱动 | 随机 | 0% ± 27.8 | 365.4 | 60% |
| `obstacles_sparse` | 麦轮 | 纯跟踪（pure pursuit） | 100% ± 27.8 | 93.1 | 0% |
| `obstacles_sparse` | 麦轮 | DWA | 100% ± 27.8 | 76.0 | 0% |
| `obstacles_sparse` | 麦轮 | 随机 | 0% ± 27.8 | 275.7 | 70% |
| `obstacles_dense` | Ackermann | 纯跟踪（pure pursuit） | 90% ± 30.4 | 84.4 | 10% |
| `obstacles_dense` | Ackermann | DWA | 100% ± 27.8 | 72.7 | 0% |
| `obstacles_dense` | Ackermann | 随机 | 0% ± 27.8 | 203.3 | 90% |
| `obstacles_dense` | 差速驱动 | 纯跟踪（pure pursuit） | 80% ± 31.0 | 79.6 | 20% |
| `obstacles_dense` | 差速驱动 | DWA | 100% ± 27.8 | 72.8 | 0% |
| `obstacles_dense` | 差速驱动 | 随机 | 0% ± 27.8 | 195.0 | 90% |
| `obstacles_dense` | 麦轮 | 纯跟踪（pure pursuit） | 80% ± 31.0 | 79.6 | 20% |
| `obstacles_dense` | 麦轮 | DWA | 100% ± 27.8 | 72.8 | 0% |
| `obstacles_dense` | 麦轮 | 随机 | 0% ± 27.8 | 242.9 | 70% |

仓库里的 `examples/eval_matrix.py` 是一次更短的 smoke 运行：它只覆盖
`obstacles_sparse`，`max_steps=300`。它的纯跟踪与 DWA 行与上表一致；random 行不同，
原因是视野（horizon）更短。要精确复现上表，请用同样的 `evaluate(...)` 协议、
`max_steps=500`，并覆盖两个场景。

```bash
python examples/eval_matrix.py
```

## 控制器

`chassis_sim.controllers` 里附带两个控制器：

| 类 | 角色 | 关键默认值 |
| --- | --- | --- |
| `PurePursuitController` | 几何路径跟踪 | `lookahead=0.6`, `adaptive=False`, `lookahead_gain=0.8` |
| `DwaController` | 基于采样的局部规划器 | `dt=0.1`, `horizon=10`, `gear_mode="off"` |

`DwaController.gear_mode="off"` 是刻意选定的默认值。在这个模式下候选轨迹横跨整个
`[v_min, v_max]` 区间，代价函数里没有挡位（前进/倒车）项；倒车交给按弧长取的前视参考
来处理。`gear_mode="auto"` 是可选消融：它从路径段推断挡位，把候选截断到半个轴，按运动
方向给航向打分，并且需要一个最低速度以免停住。它对 `heading_lookahead` 敏感，所以不是
默认值。

### 自定义控制器

`evaluate()` 不导入任何具体控制器，只需要下面这套鸭子类型协议：

| 成员 | 必需 | 契约 |
| --- | --- | --- |
| `reset()` | 是 | 重置内部状态 |
| `set_path(path_world)` | 是 | 世界系 `(N, 3)` 的 `[x, y, θ]`；`θ` 是车头朝向 |
| `act(observation)` | 是 | 返回一个归一化动作 |
| `to_chassis_action(v, omega, vy=0.0)` | 否 | 内置控制器使用的辅助方法 |

传给 `act()` 的观测 dict 含 `pose`（3 个值）、可选的 `obstacles` 与可选的 `dt`。
`chassis_sim.controllers` 里的 `Controller` 基类已经替你实现了 `to_chassis_action()`。
`examples/eval_matrix.py` 里有个很小的 `RandomController`，只实现 `reset()`、
`set_path()` 与 `act()`；它是最小的集成范例。本包不附带 MPC 控制器。

控制器必须与评测环境用同一台底盘构造，通常写成
`PurePursuitController(chassis=env.chassis)`。若两边的限制不同，`evaluate()` 会在开跑
之前抛 `ValueError`：否则控制器按一套限制归一化、`Chassis.command()` 又按另一套反归一
化，每条指令都会被静默缩放。实际消息如下。

```python
import contextlib
import io

import numpy as np

from chassis_sim.chassis import make_chassis
from chassis_sim.controllers import DwaController, PurePursuitController
from chassis_sim.envs import ChassisSimEnv
from chassis_sim.eval import controller_observation, evaluate

env = ChassisSimEnv("ackermann", max_steps=50)
obs, info = env.reset(seed=0)
observation = controller_observation(env)

pp = PurePursuitController(chassis=env.chassis)
pp.reset()
pp.set_path(env.bridge.get_path_global())
print("pure_pursuit act:", pp.act(observation))

dwa = DwaController(chassis=env.chassis, dt=env.dt)
dwa.reset()
dwa.set_path(env.bridge.get_path_global())
print("dwa act:", dwa.act(observation), "gear_mode:", dwa.gear_mode)


class ConstantController:
    def reset(self):
        pass

    def set_path(self, path_world):
        self.path = np.asarray(path_world)

    def act(self, observation):
        return np.array([0.0, 0.2], dtype=np.float32)


custom = ConstantController()
with contextlib.redirect_stderr(io.StringIO()):
    result = evaluate(
        lambda: ChassisSimEnv("ackermann", max_steps=100),
        custom,
        n_episodes=2,
        seed=0,
    )
print("custom controller:", f"success={result.success_rate:.0%}",
      f"steps={result.mean_steps:.1f}")

mismatch = PurePursuitController(chassis=make_chassis("diff_drive"))
try:
    evaluate(lambda: ChassisSimEnv("ackermann", max_steps=50), mismatch, n_episodes=1)
except ValueError as exc:
    print("ValueError:", exc)
```

输出：

```text
pure_pursuit act: [0. 1.]
dwa act: [0. 1.] gear_mode: off
custom controller: success=0% steps=100.0
ValueError: 控制器与 env 的底盘不一致 —— 控制器是按另一台底盘标定的：
  controller:  kind='diff_drive' v_min=-1.0 v_max=1.0 omega_max=1.0 vy_max=0.0
  env.chassis: kind='ackermann' v_min=-0.6 v_max=1.2 omega_max=4.804604297653527 vy_max=0.0
  不一致项: kind: 控制器 'diff_drive' vs env 'ackermann'; v_min: 控制器 -1.0 vs env -0.6; v_max: 控制器 1.0 vs env 1.2; omega_max: 控制器 1.0 vs env 4.804604297653527
  后果: to_chassis_action 按控制器的 limits 归一化、Chassis.command() 按
  env 底盘的 limits 反解 ⇒ 速度/角速度语义被静默缩放（v_max 1.0→1.2 即
  +20% 超速）。
  修法: 用 env 的底盘构造控制器（PurePursuitController(
  chassis=env.unwrapped.chassis)），或显式 evaluate(..., allow_mismatch=True)。
```

## 传感器

传感器声明为一组不可变的 spec：你需要哪些就构造哪些，连同 seed 一起传给环境（或传给
`SimpleBridge`）。

```python
from chassis_sim.sensors import Lidar, Imu, WheelOdom

sensors = [
    Lidar("scan", mount=(0.0, 0.0, 0.0), fov=3.141592653589793, n_rays=180,
          max_range=10.0, range_min=0.05, noise_std=0.0, quant_step=0.0,
          latency_steps=0, dropout_prob=0.0),
    Imu("imu", noise_std=0.0),
    WheelOdom("wheel", track_width=None, noise_std=0.0),
]
```

| Spec | 字段（默认值） |
| --- | --- |
| `Lidar` | `name`, `mount=(x, y, yaw)` `(0, 0, 0)`, `fov=π`, `n_rays=180`, `max_range=10.0`, `range_min=0.05`, `noise_std=0.0`, `quant_step=0.0`, `latency_steps=0`, `dropout_prob=0.0` |
| `Imu` | `name`, `noise_std=0.0` |
| `WheelOdom` | `name`, `track_width=None`（回退到车体宽度）, `noise_std=0.0` |

以 `sensors=[...], sensor_seed=...` 传给 `ChassisSimEnv` 或 `SimpleBridge`。
`sensors=` 与显式的 `bridge=` 互斥：传感器集合归环境所有，这样环境才能注入自己的射线
投射器。

读数进入 `reset()` 与 `step()` 的 `info` dict，默认不会自行进入观测：

| 键 | 值 |
| --- | --- |
| `info["scan"]` | `{"scan": (n_rays,) float32 or None, "scan_meta": {...}}` |
| `info["imu"]` | `{"ax", "ay", "yaw_rate", "yaw", "t"}` |
| `info["wheel"]` | `{"v_left", "v_right", "v", "t"}` |

`reset()` 报出的是桥接器在自己 reset 结束时产生的那一帧的读数，所以
`obs, info = env.reset()` 里已经带着 `scan` / `imu` / `wheel`；没有配置传感器时 `info`
保持 `{}`。LiDAR 的处理顺序是固定的：先截断到 `[range_min, range_max]`，加高斯噪声，
再截断一次，量化，转成 `float32`，推过延迟环，最后在投递时施加 dropout。因此延迟帧携带
的是它采集那一拍的噪声，而被丢掉的帧正是本该投递的那一帧。

每个传感器都有由 `(sensor_seed, name)` 派生的独立随机流，所以增删一个传感器不会挪动
其它传感器的噪声。传感器从不碰全局的 `numpy.random` 流，也不改变观测 —— 除非你要了
一个 `Lidar` 观测段（见下一节）。

```python
import numpy as np

from chassis_sim.envs import ChassisSimEnv
from chassis_sim.sensors import Imu, Lidar, WheelOdom

SENSORS = [
    Lidar("scan", n_rays=90, fov=np.pi / 2, max_range=6.0,
          noise_std=0.01, quant_step=0.05, latency_steps=2),
    Imu("imu", noise_std=0.02),
    WheelOdom("wheel", noise_std=0.01),
]
env = ChassisSimEnv("ackermann", max_steps=50, sensors=SENSORS, sensor_seed=7)
obs, info = env.reset(seed=0)
print("obs:", obs.shape, "sensor keys at reset:", sorted(info.keys()))
for _ in range(3):
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.3], dtype=np.float32)
    )
print("imu:", info["imu"])
print("wheel:", info["wheel"])
print("scan:", info["scan"]["scan"].shape, info["scan"]["scan"].dtype)
print("scan_meta:", info["scan"]["scan_meta"])

plain = ChassisSimEnv("ackermann", max_steps=50)
plain_obs, plain_info = plain.reset(seed=0)
print("default sensor keys at reset:", sorted(plain_info.keys()))
print("obs shapes with/without sensors:", obs.shape, plain_obs.shape)

sensor_env = ChassisSimEnv("ackermann", max_steps=50, sensors=SENSORS, sensor_seed=7)
plain_env = ChassisSimEnv("ackermann", max_steps=50)
sensor_obs, _ = sensor_env.reset(seed=3)
plain_obs, _ = plain_env.reset(seed=3)
identical = np.array_equal(sensor_obs, plain_obs)
for _ in range(5):
    action = np.array([0.0, 0.3], dtype=np.float32)
    sensor_obs, *_ = sensor_env.step(action)
    plain_obs, *_ = plain_env.step(action)
    identical = identical and np.array_equal(sensor_obs, plain_obs)
print("obs bit-identical with/without sensors over 5 steps:", identical)
```

输出：

```text
obs: (128,) sensor keys at reset: ['imu', 'scan', 'wheel']
imu: {'ax': -0.019766656395704817, 'ay': -0.012634059796588477, 'yaw_rate': -0.03938493375340822, 'yaw': 0.0, 't': 0.30000000000000004}
wheel: {'v_left': 0.3596476518539975, 'v_right': 0.36967108680392974, 'v': 0.3646593693289636, 't': 0.30000000000000004}
scan: (90,) float32
scan_meta: {'name': 'scan', 'mount': (0.0, 0.0, 0.0), 'n_rays': 90, 'fov': 1.5707963267948966, 'angle_min': -0.7853981633974483, 'angle_max': 0.7853981633974483, 'angle_increment': 0.017649396930279736, 'range_min': 0.05, 'range_max': 6.0, 'latency_steps': 2, 't': 0.30000000000000004, 'frame_t': 0.1, 'valid': True, 'dropped': False}
default sensor keys at reset: []
obs shapes with/without sensors: (128,) (128,)
obs bit-identical with/without sensors over 5 steps: True
```

### 缺帧与 LiDAR 观测段

`LidarSegment` 观测段（来自 `chassis_sim.obs`）把具名传感器的扫描发布进观测向量；
`chassis_sim.sensors.Lidar` 就是产生这份扫描的传感器 spec。桥接器在 `reset()` 之后立刻
产出一帧真实数据，reset 的观测带的就是这一帧。

当某一帧缺失时 —— 还在延迟环里的延迟帧，或被丢弃的帧 —— 该段保持**最后一个有效扫描**
（零阶保持）。若从未投递过有效帧，该段用 `max_range` 填满。`env.reset()` 会清空保持
缓存。观测本身不携带有效位：`info` 里的 `scan_meta["valid"]` 与
`scan_meta["dropped"]` 是唯一说明该帧曾缺失的地方。

```python
import numpy as np

from chassis_sim.envs import ChassisSimEnv
from chassis_sim.obs import Act, Goal, LidarSegment, ObsSpec, Odom, Path
from chassis_sim.scenarios import load_scenario, scenario_to_reset_options
from chassis_sim.sensors import Lidar

SPEC = ObsSpec((
    Odom(),
    Path(points=30),
    Goal(),
    Act(action_dim=2),
    LidarSegment("scan", sensors=("scan",)),
))

empty = ChassisSimEnv(
    "ackermann", max_steps=5, obs_spec=SPEC,
    sensors=[Lidar("scan", n_rays=36, max_range=3.0, dropout_prob=1.0)],
    sensor_seed=0,
)
obs, info = empty.reset(seed=0)
span = empty.obs_spec_describe()["scan"]
print("describe:", empty.obs_spec_describe())
obs, reward, terminated, truncated, info = empty.step(
    np.array([0.0, 0.4], dtype=np.float32)
)
print("dropout_prob=1.0 -> unique values in the scan segment:",
      np.unique(obs[span]), "meta.valid:", info["scan"]["scan_meta"]["valid"])

hold = ChassisSimEnv(
    "ackermann", max_steps=40, obs_spec=SPEC,
    sensors=[Lidar("scan", n_rays=36, max_range=3.0, dropout_prob=0.6)],
    sensor_seed=5,
)
obs, info = hold.reset(seed=0, options=scenario_to_reset_options(load_scenario("obstacles_dense")))
previous = obs[span].copy()
rows = []
for step in range(1, 21):
    obs, reward, terminated, truncated, info = hold.step(
        np.array([0.0, 0.4], dtype=np.float32)
    )
    meta = info["scan"]["scan_meta"]
    rows.append((step, bool(meta["valid"]), bool(np.array_equal(obs[span], previous))))
    previous = obs[span].copy()
dropped = [(step, same) for step, valid, same in rows if not valid]
valid = [(step, same) for step, valid, same in rows if valid]
print("dropped steps:", len(dropped), "all held from the previous frame:",
      all(same for _, same in dropped))
print("valid steps:", len(valid), "of which the segment moved:",
      sum(1 for _, same in valid if not same))
print("scan range in obs:", float(obs[span].min()), float(obs[span].max()))
```

输出：

```text
describe: {'odom': slice(0, 4, None), 'path': slice(4, 124, None), 'goal': slice(124, 126, None), 'act': slice(126, 128, None), 'scan': slice(128, 164, None)}
dropout_prob=1.0 -> unique values in the scan segment: [3.] meta.valid: False
dropped steps: 13 all held from the previous frame: True
valid steps: 7 of which the segment moved: 7
scan range in obs: 0.5022975206375122 3.0
```

## 动力学

执行器模型分三档。用构造参数或 `EnvConfig.dynamics` 选择；构造参数覆盖配置。

| 档位 | 语义 | 默认值 |
| --- | --- | --- |
| `none` | 纯运动学；`dynamics.py` 不被导入也不被调用；没有 `info["dynamics"]` 键 | — |
| `limits` | 加速度与转向速率限制 | `accel_max = 1.5 m/s²`, `steer_rate_max = 3.0 rad/s` |
| `wheel` | 在限制之上再加一阶轮速滞后、死区与打滑 | `v_tau = 0.15 s`, `omega_tau = 0.10 s`, `v_deadband = 0.0`, `wheel_slip = 0.0` |

`EnvConfig.accel_max` 与 `EnvConfig.steer_rate_max` 默认是 `0.0`。对 `limits` 与
`wheel`，非正值表示“未设置”，会解析为上面那组 1.5 m/s² / 3.0 rad/s 默认值；`none` 则
原样保留配置值。`wheel` 施加 `v += (v_cmd - v) * (1 - exp(-dt / v_tau))`，`ω` 走同样的
一阶更新，然后再乘以 `(1 - wheel_slip)`；落在 `v_deadband` 内的指令被强制归零。

`none` 这条路径与引入本开关之前的行为逐位相同：不导入 dynamics 模块、不调用它、也不
消耗随机数。下面的片段在 20 步内比对 `dynamics=None` 与 `dynamics="none"`，打印
`True`。它通过 `obs_spec_describe()` 读里程计，而不是硬编码切片。

```python
import numpy as np

from chassis_sim.envs import ChassisSimEnv, EnvConfig


def run(level, **overrides):
    cfg = EnvConfig(dynamics=level, **overrides)
    env = ChassisSimEnv("diff_drive", cfg=cfg, max_steps=10)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    return obs[env.obs_spec_describe()["odom"]], info.get("dynamics")


for level in ("none", "limits", "wheel"):
    odom, dynamics_info = run(level)
    print(f"{level:6s} odom={np.round(odom, 6)} info.dynamics={dynamics_info}")

odom, dynamics_info = run("wheel", wheel_slip=0.2)
print(f"wheel + slip=0.2 odom={np.round(odom, 6)} info.dynamics={dynamics_info}")

default_env = ChassisSimEnv("diff_drive", max_steps=20)
none_env = ChassisSimEnv("diff_drive", dynamics="none", max_steps=20)
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

输出：

```text
none   odom=[1.2 0.  1.2 0. ] info.dynamics=None
limits odom=[0.15 0.   0.15 0.  ] info.dynamics={'level': 'limits', 'accel_max': 1.5, 'steer_rate_max': 3.0}
wheel  odom=[0.072987 0.       0.072987 0.      ] info.dynamics={'level': 'wheel', 'accel_max': 1.5, 'steer_rate_max': 3.0, 'v_tau': 0.15, 'omega_tau': 0.1, 'alpha_v': 0.486582880967408, 'alpha_omega': 0.6321205588285577, 'v_deadband': 0.0, 'wheel_slip': 0.0}
wheel + slip=0.2 odom=[0.05839 0.      0.05839 0.     ] info.dynamics={'level': 'wheel', 'accel_max': 1.5, 'steer_rate_max': 3.0, 'v_tau': 0.15, 'omega_tau': 0.1, 'alpha_v': 0.486582880967408, 'alpha_omega': 0.6321205588285577, 'v_deadband': 0.0, 'wheel_slip': 0.2}
dynamics=None vs 'none' obs bit-identical over 20 steps: True
```

## 规划

`pip install "chassis-sim[planning]"` 会装上 `numba`。该 extra 提供 `MapGenerator`、
`hybrid_astar_search`，以及 Reeds-Shepp 的基本元（`reeds_shepp_path`、
`reeds_shepp_path_pose`、`reeds_shepp_length`）。

cusp 过滤器会拒掉那些倒车折返太短、或离前进段太近的参考路径。默认值取自
`config/curriculum.yaml`：

| 键 | 默认值 | 作用 |
| --- | ---: | --- |
| `min_cusp_fold_arc` | 0.8 m | 拒掉短于该值的极大倒车段 |
| `min_cusp_leg_separation` | 0.35 m | 拒掉与前进段距离小于该值的倒车段 |
| `cusp_min_arc_gap` | 1.0 m | 只比较弧长间隔不小于该值的段对 |

把某个阈值设为 `0` 即可关闭该判据。被拒的候选会重新采样；
`MapGenerator.cusp_rejections` 统计被拒次数。

没有 numba 时核心场景照常可跑；规划类场景会抛 `ImportError`，消息里带安装提示：

```python
import sys

from chassis_sim.envs import ChassisSimEnv
from chassis_sim.planning import MapGenerator, hybrid_astar_search, reeds_shepp_length
from chassis_sim.scenarios import scenario_reset_options_factory

print("planning imports:",
      MapGenerator.__name__, hybrid_astar_search.__name__, reeds_shepp_length.__name__)

factory = scenario_reset_options_factory("obstacles_sparse", base_seed=0)
env = ChassisSimEnv("ackermann", max_steps=100)
obs, info = env.reset(seed=0, options=factory(0))
print("generated map:",
      f"obstacles={len(env._obstacles)}",
      f"path={env.bridge.get_path_global().shape}",
      f"factory.stats={factory.stats}")

sys.modules["numba"] = None
try:
    factory_without_numba = scenario_reset_options_factory("obstacles_sparse", base_seed=0)
    factory_without_numba(0)
except ImportError as exc:
    print("ImportError:", exc)
```

输出：

```text
planning imports: MapGenerator hybrid_astar_search reeds_shepp_length
generated map: obstacles=5 path=(152, 3) factory.stats={'attempts': 1, 'retries': 0, 'failures': 0, 'episodes': 1}
ImportError: 场景 'obstacles_sparse' 需要 Hybrid-A* 地图生成器（chassis_sim.planning），它依赖 numba。请先安装 planning extra：pip install chassis-sim[planning]
```

## 渲染

`chassis_sim.render` 用 matplotlib 的 `Agg` 后端画当前桥接器状态，因此可以无头运行。
`render_to_png()` 写一帧；`render_episode_frames()` 让桥接器走过一段动作序列并写出各帧
（总是包含最后一步）；`save_gif()` 把帧组装起来。matplotlib 是懒导入的，只在渲染时
需要。`imageio` 可选：缺了它，`save_gif()` 会写一个帧目录并打一条警告，而不是抛异常。
`examples/gallery.py` 就建在同一组原语（`render_frame`、`save_figure`、`save_gif`）之上，
产出图库一节里的素材。

```python
from pathlib import Path

import numpy as np

from chassis_sim.envs import ChassisSimEnv
from chassis_sim.render import render_episode_frames, render_to_png

env = ChassisSimEnv("ackermann", max_steps=50)
obs, info = env.reset(seed=0)
out_dir = Path("/tmp/chassis_sim_readme_render")
out_dir.mkdir(parents=True, exist_ok=True)

png = render_to_png(env.bridge, out_dir / "frame.png", dpi=80)
print("png:", png.name, "exists:", png.exists())

actions = [
    np.array([0.2 * np.sin(index / 5.0), 0.4], dtype=np.float64)
    for index in range(10)
]
frames = render_episode_frames(env.bridge, actions, out_dir, every=5, prefix="step", dpi=80)
print("frames:", [frame.name for frame in frames])
print("all non-empty:", all(frame.stat().st_size > 0 for frame in frames))
```

输出：

```text
png: frame.png exists: True
frames: ['step_00005.png', 'step_00010.png']
all non-empty: True
```

## 示例

| 脚本 | 作用 |
| --- | --- |
| `examples/dynamic_demo.py` | `dynamic_crossing` 的静态与动态对比，跑 20 步；验证在没有动态障碍时回放逐位一致 |
| `examples/eval_matrix.py` | 稀疏 smoke 运行：3 个控制器 × 3 台底盘 × 10 张独立地图，`max_steps=300`；写出 Markdown 与 JSON |
| `examples/gallery.py` | 重新生成十一个图库素材（场景 seed 0 下的五段动画、六张场景静态图） |
| `examples/render_demo.py` | 50 步、5 张 PNG 帧，以及七轮渲染计时 |
| `examples/scenario_demo.py` | `straight`、`corridor` 与 `dense_static` 各跑 50 个随机步 |
| `examples/sensors_demo.py` | IMU、轮式里程计，以及带延迟与 dropout 的参数化 LiDAR |

在仓库根目录、用装着本包的那个解释器运行它们。

## 已知限制

1. **离散碰撞检测，没有扫掠体。** 碰撞是对当前车体 OBB 与圆形、带朝向矩形（含安全余
   量）的逐拍离散检测；没有扫掠体／连续（CCD）检测，所以快速倒车仍可能在两拍之间擦到
   拐角。
2. **观测里缺失的 LiDAR 帧是零阶保持，不是真值。** 帧被延迟或丢弃时，`Lidar` 观测段
   重复最后一个有效扫描（首个有效帧之前回退到 `max_range`）。观测不携带有效位，消费
   方不读 `info` 里的 `scan_meta` 就无法区分新扫描与保持帧。
3. **DWA 的挡位选择默认关闭。** `gear_mode="auto"` 会把候选截断到单一方向，且对
   `heading_lookahead` 敏感；配合上面的离散碰撞检测，一次硬性挡位选择可能在某个机动中
   碰撞，而全区间候选本可避开。
4. **`get_scan()` 的元数据是历史遗留。** 这个便利辅助返回十个经典 LaserScan 键，常量
   固定为 180 条射线、π 视场（FOV）、10 m 最大量程、0.05 m 最小量程。参数化的 LiDAR
   设置（射线数、FOV、量程、延迟、dropout）只在 `info["scan"]["scan_meta"]` 中报出。
5. **没有相机。** 不含 RGB、深度或分割传感器。
6. **传感器只有被请求时才进入观测。** IMU 与轮式里程计只通过 `info` 报出，LiDAR 也只
   有当某个 `LidarSegment` 点名它时才并入观测向量。
7. **核心场景只有一张地图。** 核心场景是固定 YAML 地图；只有规划类场景（以及自定义
   callable）才会每回合产出一张新地图。
8. **图库动画是 10 Hz 循环的 12 fps 抽样。** 动画在帧之间跳过若干拍，播放快于实时，
   所以它只是行为的示意，不是控制序列的记录；要按当前代码重建这些帧，请重跑
   `examples/gallery.py`。

## 许可证、贡献与引用

MIT。见 `LICENSE`。

测试与 lint：

```bash
pytest -q
ruff check .
```

CI 在 Python 3.10、3.11、3.12 上跑测试套件（`.github/workflows/ci.yml`）。

若在学术工作中使用本包，请引用本仓库：

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

本 README 中的所有数字均于 2026-11-04（UTC）在 Python 3.12.0、NumPy 1.26.4、
Gymnasium 1.3.0 上测得，对应 0.2.1 版本。默认观测在两台 2 维动作底盘上是 128 维、在
麦轮上是 129 维。评测矩阵使用十张独立地图、`max_steps=500` 与 `base_seed=0`。本文件中
的代码块在仓库的校验流程里是按原样执行的。
