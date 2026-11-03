# chassis-sim

[![CI](https://github.com/zhao6663/chassis-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/zhao6663/chassis-sim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/zhao6663/chassis-sim)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A small, dependency-light Gymnasium benchmark for three chassis types** — Ackermann
steering, differential drive, and mecanum — with normalized actions, a pure-NumPy
simulator, optional parameterized sensors, and reference controllers for reproducible
evaluation.

Version 0.1.4. Python 3.10+.

## At a glance

Every environment exposes a normalized action space, `Box(-1, 1, (d,), float32)`.
The chassis turns that normalized action into physical quantities; the environment
never asks a user to supply volts, wheel torques, or raw steering angles.

| Gymnasium id | Chassis | Action dim | Action layout | Physical limits with default config |
| --- | --- | ---: | --- | --- |
| `ChassisSim-Ackermann-v0` | Ackermann | 2 | `[steer, speed]` | `v ∈ [-0.6, 1.2] m/s`, `δ ∈ [-0.526, 0.526] rad`, `|ω| ≤ 4.805 rad/s` |
| `ChassisSim-DiffDrive-v0` | differential drive | 2 | `[speed, yaw]` | `v ∈ [-1.2, 1.2] m/s`, `|ω| ≤ 1.0 rad/s` |
| `ChassisSim-Mecanum-v0` | mecanum | 3 | `[vx, vy, yaw]` | `v_x ∈ [-1.2, 1.2] m/s`, `|vy| ≤ 0.5 m/s`, `|ω| ≤ 1.0 rad/s` |

The limits above are what `env.chassis.describe()["limits"]` reports with the
default `EnvConfig`; they are not duplicated in the environment. A different
`EnvConfig` (or a different `Chassis` instance) changes the physical mapping and
the action semantics together. For the Ackermann chassis, `delta_max` and
`curvature_action` decide whether the first action axis means steering angle or
curvature.

## Installation

From PyPI:

```bash
pip install chassis-sim
```

From a checkout, with the development tools:

```bash
git clone https://github.com/zhao6663/chassis-sim.git
cd chassis-sim
pip install -e ".[dev]"
```

### Extras

| Extra | Installs | Purpose |
| --- | --- | --- |
| `planning` | `numba>=0.56` | Hybrid-A* map generation for the planning scenarios |
| `viz` | `matplotlib>=3.7`, `rich>=13.0` | rendering and rich terminal helpers |
| `dev` | `pytest>=7.0`, `ruff>=0.4` | test suite and linter |

The core install depends only on `numpy`, `gymnasium`, and `pyyaml`.
`matplotlib` is needed only for rendering; `imageio` is optional and only used
for GIF export.

The 0.1.x line ships exactly one built-in simulator: the pure-NumPy `SimpleBridge`. No
external simulator is required. The ir-sim and ROS 2 backends are not part of this
package (they remain in the upstream training repository).

## Quickstart

```python
import gymnasium as gym
import numpy as np

import chassis_sim  # noqa: F401  (registers the three environment ids)

env = gym.make("ChassisSim-Ackermann-v0", max_steps=100)
obs, info = env.reset(seed=0)
print("obs:", obs.shape, obs.dtype)
print("action_space:", env.action_space)

total_reward = 0.0
steps = 0
terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.3], dtype=np.float32)
    )
    total_reward += float(reward)
    steps += 1

print(f"steps={steps} total_reward={total_reward:.3f}")
print(f"terminated={terminated} truncated={truncated}")
print("termination_reason:", info["termination_reason"])
print("info:", {key: info[key] for key in ("step", "collision", "goal_reached", "d_min")})
env.close()
```

Running it prints:

```text
obs: (494,) float32
action_space: Box(-1.0, 1.0, (2,), float32)
steps=100 total_reward=-28.756
terminated=False truncated=True
termination_reason: timeout
info: {'step': 100, 'collision': False, 'goal_reached': False, 'd_min': 4.445171356201172}
```

The three registered ids are `ChassisSim-Ackermann-v0`,
`ChassisSim-DiffDrive-v0`, and `ChassisSim-Mecanum-v0`. They all use the same
observation builder, reward, and termination logic; only the chassis, action width,
and the `act` slice of the observation change.

`ChassisSimEnv` terminates on its own `max_steps`; `gym.make` does not add a
`TimeLimit` wrapper (`max_episode_steps=None` at registration). Pass
`max_steps=...` to shorten an episode.

## Observation and action contract

| Chassis | Observation dim | Why |
| --- | ---: | --- |
| Ackermann | 494 | `prev_action` carries 2 values |
| differential drive | 494 | `prev_action` carries 2 values |
| mecanum | 495 | `prev_action` carries 3 values |

The observation is a single `float32` vector. Its `Box` bounds are
`(-inf, inf)`; the values below are already in the units the network sees.

| Segment | Width | Slice, `d=2` | Slice, `d=3` | Meaning |
| --- | ---: | --- | --- | --- |
| `lidar` | 360 | `[0:360]` | `[0:360]` | 360 LiDAR ranges (front 180 + rear 180) |
| `odom` | 4 | `[360:364]` | `[360:364]` | body-frame `[dx*10, dy*10, v, ω]` |
| `path` | 120 | `[364:484]` | `[364:484]` | 30 points × `[rho, sin_phi, cos_phi, dtheta]` |
| `goal` | 2 | `[484:486]` | `[484:486]` | goal position in the body frame |
| `act` | `d` | `[486:488]` | `[486:489]` | previous normalized action |
| `err` | 2 | `[488:490]` | `[489:491]` | normalized cross-track error + heading error |
| `goal_sig` | 2 | `[490:492]` | `[491:493]` | `sin` / `cos` of the goal heading error |
| `path_valid` | 1 | `[492:493]` | `[493:494]` | fraction of real path points, `[0, 1]` |
| `path_progress` | 1 | `[493:494]` | `[494:495]` | arc completion `[0, 1]`, monotone non-decreasing |

The action is always `Box(-1, 1, (d,), float32)`. The normalized values become
physical commands inside the chassis: `Chassis.command()` maps an action to
`(v, ω, vy)`, and `Chassis.step()` advances the kinematic state. For the
Ackermann chassis, `step()` uses the previous steering angle for the yaw rate and
then stores the new one, so there is a one-tick steering lag by construction.

Print the layout from code:

```python
import gymnasium as gym
import numpy as np

import chassis_sim  # noqa: F401
from chassis_sim.envs import ObsLayout

for env_id in ("ChassisSim-Ackermann-v0", "ChassisSim-DiffDrive-v0", "ChassisSim-Mecanum-v0"):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    print(f"{env_id:26s} obs={obs.shape} {obs.dtype} action_dim={env.action_space.shape[0]}")
    env.close()

for action_dim in (2, 3):
    layout = ObsLayout(action_dim=action_dim, max_path_points=30)
    parts = layout.slice(np.zeros(layout.total_dim, dtype=np.float32))
    offset = 0
    print(f"action_dim={action_dim} total={layout.total_dim}")
    for name in layout.ORDER:
        width = parts[name].shape[-1]
        print(f"  {name:14s} [{offset}:{offset + width}]")
        offset += width
```

Output:

```text
ChassisSim-Ackermann-v0    obs=(494,) float32 action_dim=2
ChassisSim-DiffDrive-v0    obs=(494,) float32 action_dim=2
ChassisSim-Mecanum-v0      obs=(495,) float32 action_dim=3
action_dim=2 total=494
  lidar          [0:360]
  odom           [360:364]
  path           [364:484]
  goal           [484:486]
  act            [486:488]
  err            [488:490]
  goal_sig       [490:492]
  path_valid     [492:493]
  path_progress  [493:494]
action_dim=3 total=495
  lidar          [0:360]
  odom           [360:364]
  path           [364:484]
  goal           [484:486]
  act            [486:489]
  err            [489:491]
  goal_sig       [491:493]
  path_valid     [493:494]
  path_progress  [494:495]
```

### Physical-command bridges

A third-party simulator (ir-sim, ROS/Gazebo, the main repo's `NumpySimBridge`) takes
**physical** commands, not a normalized action. `PhysicalCommandBridge` maps with
`chassis.physical_action()` (never `step()` — the backend owns its own steering lag)
and forwards sensors untouched. The vector order is the chassis's own:

| chassis | physical vector |
| --- | --- |
| `ackermann` | `[delta (rad), v (m/s)]` |
| `diff_drive` | `[v (m/s), omega (rad/s)]` |
| `mecanum` | `[v (m/s), vy (m/s), omega (rad/s)]` |

The actuator lag must be applied **exactly once**, by the inner physical simulator:
the adapter never advances the chassis state and never calls `chassis.reset_state()`.
An inner bridge that does not model its own lag drops the pose equivalence with
`SimpleBridge` from 1.5e-15 to 9.76e-01.

```python
from chassis_sim.bridges import PhysicalCommandBridge
from chassis_sim.chassis import AckermannChassis
ch = AckermannChassis()                    # the same instance the env uses
bridge = PhysicalCommandBridge(inner, ch)  # inner speaks [delta, v] (this chassis)
bridge.step([0.3, 0.6])                    # -> inner.step(array([delta, v]))
```

## Scenarios

Eleven YAML scenarios ship with the package. The first five are core scenarios and
need only the base install. The last six use Hybrid-A* and Reeds-Shepp planning and
declare `requires: planning`.

| Scenario | Requires | Seed | Static obstacles | Notes |
| --- | --- | ---: | ---: | --- |
| `circle` | core | 0 | 0 | circular reference path |
| `corridor` | core | 0 | 2 | narrow corridor |
| `dense_static` | core | 0 | 14 | dense static field |
| `dynamic_crossing` | core | 0 | 0 | moving obstacles supplied through reset options |
| `straight` | core | 0 | 0 | straight reference path |
| `obstacles_sparse` | `planning` | 0 | 0 | generated sparse field |
| `obstacles_medium` | `planning` | 0 | 0 | generated medium field |
| `obstacles_dense` | `planning` | 0 | 0 | generated dense field |
| `obstacles_dense_reverse` | `planning` | 0 | 0 | dense field with reverse-oriented paths |
| `foldback_reverse` | `planning` | 0 | 0 | reverse folds, cusp-filtered |
| `maze_grid` | `planning` | 1 | 0 | grid maze (YAML seed 1) |

Load a scenario once, or build a per-episode reset-options factory:

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

Output:

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
corridor reset: (494,) obstacles= 2 path= (141, 3)
episode=0 obstacles=5 path=(152, 3)
episode=1 obstacles=4 path=(168, 3)
factory.stats: {'attempts': 2, 'retries': 0, 'failures': 0, 'episodes': 2}
```

`scenario_to_reset_options(scenario)` returns a plain dict. Reusing that dict for
every episode replays the same map; the evaluation section below explains why that
matters. `scenario_reset_options_factory(name, base_seed=0)` returns a callable that
generates a fresh map for each episode index and exposes its bookkeeping in
`factory.stats`.

## Evaluation

`evaluate()` runs a controller for `n_episodes` and returns an `EvalResult`.
`wilson_interval(successes, n)` returns the Wilson score interval at 95% by default
(`z = 1.959963984540054`).

```text
evaluate(env_factory, controller, *, n_episodes, seed=0, max_steps=None,
         reset_options=None, labels=None, allow_mismatch=False) -> EvalResult
wilson_interval(successes, n, *, z=1.959963984540054) -> (low, high)
```

`EvalResult` exposes `n`, `successes`, `collisions`, `timeouts`,
`success_rate`, `collision_rate`, `timeout_rate`, `mean_steps`,
`mean_return`, `min_obstacle_clearance`, `success_interval`, `success_ci`,
`map_fingerprints`, `distinct_maps`, `maps_independent`, and
`same_map_repeated`. `success_ci` is `max(success_rate - low, high - success_rate)`,
the larger Wilson half-width; the interval itself is asymmetric.

- `distinct_maps` is the number of distinct map fingerprints in the run. `0` means
  the maps could not be fingerprinted.
- `maps_independent` is `True` when `n <= 1`, or when every episode has a
  non-empty fingerprint and all fingerprints are distinct.
- `same_map_repeated` is `True` when the run contains a repeated map.

A fixed reset-options dict is accepted, but it replays one map. The evaluator warns
on stderr and the confidence interval must not be used as an independent-sample CI.
Two episodes with a fixed dict print:

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

Output:

```text
distinct_maps: 1
maps_independent: False
same_map_repeated: True
[WARN][eval] reset_options 是固定 dict：2 个回合跑的是**同一张地图**（回合 seed 只推进 env 内部 RNG，不重掷地图）⇒ success_ci 只是该图上的重放方差，不代表独立样本，不可用于控制器对比。要每回合一张独立地图请传 callable(episode_index)（例：scenario_reset_options_factory('obstacles_dense', base_seed=0)）。
```

Pass a callable instead, as in the example below. This is the difference between a
CI over ten independent maps and a CI over ten replays of one map.

```python
import numpy as np

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

Output:

```text
success=100.0% interval=(72.2%, 100.0%) steps=94.3 collisions=0.0%
distinct_maps=10 maps_independent=True same_map_repeated=False
wilson 10/10 -> [0.7225, 1.0000]
wilson 9/10 -> [0.5958, 0.9821]
wilson 8/10 -> [0.4902, 0.9433]
wilson 0/10 -> [0.0000, 0.2775]
```

### Reference matrix

The table below was produced by exactly this protocol: `n=10` independent maps,
`max_steps=500`, `base_seed=0`, Wilson 95% CI. Success is shown as
`rate ± success_ci` (the larger Wilson half-width); mean steps and collision rate
cover all ten episodes. Every
cell reports `distinct_maps=10` and `maps_independent=True`. The random baseline
samples uniformly from the normalized action space.

| Scenario | Chassis | Controller | Success (Wilson 95%) | Mean steps | Collision rate |
| --- | --- | --- | ---: | ---: | ---: |
| `obstacles_sparse` | Ackermann | pure pursuit | 100% ± 27.8 | 94.3 | 0% |
| `obstacles_sparse` | Ackermann | DWA | 100% ± 27.8 | 76.9 | 0% |
| `obstacles_sparse` | Ackermann | random | 0% ± 27.8 | 303.3 | 70% |
| `obstacles_sparse` | differential | pure pursuit | 100% ± 27.8 | 93.1 | 0% |
| `obstacles_sparse` | differential | DWA | 100% ± 27.8 | 76.0 | 0% |
| `obstacles_sparse` | differential | random | 0% ± 27.8 | 365.4 | 60% |
| `obstacles_sparse` | mecanum | pure pursuit | 100% ± 27.8 | 93.1 | 0% |
| `obstacles_sparse` | mecanum | DWA | 100% ± 27.8 | 76.0 | 0% |
| `obstacles_sparse` | mecanum | random | 0% ± 27.8 | 275.7 | 70% |
| `obstacles_dense` | Ackermann | pure pursuit | 90% ± 30.4 | 84.4 | 10% |
| `obstacles_dense` | Ackermann | DWA | 100% ± 27.8 | 72.7 | 0% |
| `obstacles_dense` | Ackermann | random | 0% ± 27.8 | 203.3 | 90% |
| `obstacles_dense` | differential | pure pursuit | 80% ± 31.0 | 79.6 | 20% |
| `obstacles_dense` | differential | DWA | 100% ± 27.8 | 72.8 | 0% |
| `obstacles_dense` | differential | random | 0% ± 27.8 | 195.0 | 90% |
| `obstacles_dense` | mecanum | pure pursuit | 80% ± 31.0 | 79.6 | 20% |
| `obstacles_dense` | mecanum | DWA | 100% ± 27.8 | 72.8 | 0% |
| `obstacles_dense` | mecanum | random | 0% ± 27.8 | 242.9 | 70% |

The shipped `examples/eval_matrix.py` is a shorter smoke run: it covers only
`obstacles_sparse` with `max_steps=300`. Its pure-pursuit and DWA rows match the
table above; its random rows differ because of the shorter horizon. To reproduce
the table exactly, use the same `evaluate(...)` protocol with `max_steps=500`
and both scenarios.

```bash
python examples/eval_matrix.py
```

## Controllers

Two controllers ship in `chassis_sim.controllers`:

| Class | Role | Key defaults |
| --- | --- | --- |
| `PurePursuitController` | geometric path tracking | `lookahead=0.6`, `adaptive=False`, `lookahead_gain=0.8` |
| `DwaController` | sampling-based local planner | `dt=0.1`, `horizon=10`, `gear_mode="off"` |

`DwaController.gear_mode="off"` is the default on purpose. In that mode candidate
trajectories span the entire `[v_min, v_max]` range and the cost has no gear
(forward/reverse) term; reversing is handled by the arc-length lookahead reference.
`gear_mode="auto"` is an optional ablation: it infers the gear from the path
segment, truncates candidates to one half-axis, scores heading against the motion
direction, and needs a minimum speed to avoid stopping. It is sensitive to
`heading_lookahead`, so it is not the default.

### Custom controllers

`evaluate()` does not import a concrete controller; it only needs the duck-typed
protocol:

| Member | Required | Contract |
| --- | --- | --- |
| `reset()` | yes | reset internal state |
| `set_path(path_world)` | yes | `(N, 3)` world-frame `[x, y, θ]`; `θ` is the nose heading |
| `act(observation)` | yes | return a normalized action |
| `to_chassis_action(v, omega, vy=0.0)` | no | helper used by the shipped controllers |

The observation dict passed to `act()` has `pose` (3 values), optional
`obstacles`, and optional `dt`. The `Controller` base class in
`chassis_sim.controllers` implements `to_chassis_action()` for you.
`examples/eval_matrix.py` contains a ~40-line `RandomController` that implements
only `reset()`, `set_path()`, and `act()`; it is the minimal integration
example. The package does not bundle an MPC controller.

Controllers must be constructed with the same chassis as the evaluation
environment, usually `PurePursuitController(chassis=env.chassis)`. If the limits
differ, `evaluate()` raises `ValueError` before running: the controller would
normalize with one set of limits and `Chassis.command()` would de-normalize with
another, silently scaling every command. The actual message is shown below.

```python
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

Output:

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

## Sensors

Sensors are off by default. When enabled they write readings to `info` only:
`info["imu"]`, `info["wheel"]`, `info["scan"]`, and `info["scan_meta"]`.
They never change the observation and never consume the global `numpy.random`
stream; each sensor set owns a `numpy.random.Generator` derived from
`SensorConfig.sensor_seed`.

| `SensorConfig` field | Default | Meaning |
| --- | --- | --- |
| `lidar` | `None` | `LidarParams` or disabled |
| `imu` | `False` | enable IMU |
| `wheel` | `False` | enable wheel odometry |
| `imu_noise_std` | `0.0` | additive noise on `ax`, `ay`, `yaw_rate` |
| `wheel_noise_std` | `0.0` | additive noise on left/right wheel speeds |
| `track_width` | `None` | wheel track; defaults to the chassis width |
| `sensor_seed` | `0` | independent random stream |

| `LidarParams` field | Default | Meaning |
| --- | ---: | --- |
| `n_rays` | 180 | rays per side |
| `fov` | `π` | field of view per side |
| `max_range` | 10.0 | maximum range (m) |
| `range_min` | 0.05 | minimum range (m) |
| `noise_std` | 0.0 | additive Gaussian range noise (m) |
| `quant_step` | 0.0 | range quantization step (m); 0 disables |
| `latency_steps` | 0 | deliver the frame collected this many steps ago |
| `dropout_prob` | 0.0 | probability of delivering `None` for a frame |
| `side` | `"front"` | mounting side (`front` or `rear`) |

LiDAR processing order is fixed: clip to `[range_min, range_max]`, add Gaussian
noise, clip again, quantize, cast to `float32`, push through the latency ring,
then apply dropout at delivery time. Delayed frames therefore carry the noise of
their acquisition step, and a dropped frame is the one that would have been
delivered.

```python
import numpy as np

from chassis_sim.bridges.sensors import LidarParams, SensorConfig
from chassis_sim.envs import ChassisSimEnv

config = SensorConfig(
    imu=True,
    wheel=True,
    lidar=LidarParams(
        n_rays=90,
        fov=np.pi / 2,
        max_range=6.0,
        noise_std=0.01,
        quant_step=0.05,
        latency_steps=2,
    ),
    sensor_seed=7,
)
env = ChassisSimEnv("ackermann", max_steps=50, sensors=config)
obs, info = env.reset(seed=0)
print("obs:", obs.shape,
      "sensor keys at reset:", [k for k in ("imu", "wheel", "scan", "scan_meta") if k in info])
for _ in range(3):
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.3], dtype=np.float32)
    )
print("imu:", info["imu"])
print("wheel:", info["wheel"])
print("scan:", info["scan"].shape, info["scan"].dtype)
print("scan_meta:", info["scan_meta"])

plain = ChassisSimEnv("ackermann", max_steps=50)
plain_obs, plain_info = plain.reset(seed=0)
print("default sensor keys:", [k for k in ("imu", "wheel", "scan", "scan_meta") if k in plain_info])
print("obs shapes with/without sensors:", obs.shape, plain_obs.shape)

sensor_env = ChassisSimEnv("ackermann", max_steps=50, sensors=config)
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

Output:

```text
obs: (494,) sensor keys at reset: []
imu: {'ax': 0.0, 'ay': 0.0, 'yaw_rate': 0.0, 'yaw': 0.0, 't': 0.30000000000000004}
wheel: {'v_left': 0.36000001430511475, 'v_right': 0.36000001430511475, 'v': 0.36000001430511475, 't': 0.30000000000000004}
scan: (90,) float32
scan_meta: {'n_rays': 90, 'fov': 1.5707963267948966, 'angle_min': -0.7853981633974483, 'angle_max': 0.7853981633974483, 'angle_increment': 0.017649396930279736, 'range_min': 0.05, 'range_max': 6.0, 'side': 'front', 'latency_steps': 2, 't': 0.30000000000000004, 'frame_t': 0.1, 'valid': True, 'dropped': False}
default sensor keys: []
obs shapes with/without sensors: (494,) (494,)
obs bit-identical with/without sensors over 5 steps: True
```

## Dynamics

The actuator model has three levels. Select one with the constructor argument or
with `EnvConfig.dynamics`; the constructor argument overrides the config.

| Level | Semantics | Defaults |
| --- | --- | --- |
| `none` | pure kinematics; `dynamics.py` is not imported or called; no `info["dynamics"]` key | — |
| `limits` | acceleration and steering-rate limits | `accel_max = 1.5 m/s²`, `steer_rate_max = 3.0 rad/s` |
| `wheel` | limits plus first-order wheel lag, deadband, and slip | `v_tau = 0.15 s`, `omega_tau = 0.10 s`, `v_deadband = 0.0`, `wheel_slip = 0.0` |

`EnvConfig.accel_max` and `EnvConfig.steer_rate_max` default to `0.0`. For
`limits` and `wheel`, a non-positive value means "unset" and resolves to the
1.5 m/s² / 3.0 rad/s defaults shown above; `none` leaves the configured values
untouched. `wheel` applies
`v += (v_cmd - v) * (1 - exp(-dt / v_tau))`, the same first-order update for
`ω`, then multiplies by `(1 - wheel_slip)`; a command inside `v_deadband` is
forced to zero.

The `none` path is bit-identical to the behavior before the switch: it does not
import the dynamics module, does not call it, and consumes no random numbers. The
snippet below checks `dynamics=None` against `dynamics="none"` over 20 steps and
prints `True`.

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
    return obs[360:364], info.get("dynamics")


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

Output:

```text
none   odom=[1.2 0.  1.2 0. ] info.dynamics=None
limits odom=[0.15 0.   0.15 0.  ] info.dynamics={'level': 'limits', 'accel_max': 1.5, 'steer_rate_max': 3.0}
wheel  odom=[0.072987 0.       0.072987 0.      ] info.dynamics={'level': 'wheel', 'accel_max': 1.5, 'steer_rate_max': 3.0, 'v_tau': 0.15, 'omega_tau': 0.1, 'alpha_v': 0.486582880967408, 'alpha_omega': 0.6321205588285577, 'v_deadband': 0.0, 'wheel_slip': 0.0}
wheel + slip=0.2 odom=[0.05839 0.      0.05839 0.     ] info.dynamics={'level': 'wheel', 'accel_max': 1.5, 'steer_rate_max': 3.0, 'v_tau': 0.15, 'omega_tau': 0.1, 'alpha_v': 0.486582880967408, 'alpha_omega': 0.6321205588285577, 'v_deadband': 0.0, 'wheel_slip': 0.2}
dynamics=None vs 'none' obs bit-identical over 20 steps: True
```

## Planning

`pip install "chassis-sim[planning]"` adds `numba`. The extra provides
`MapGenerator`, `hybrid_astar_search`, and the Reeds-Shepp primitives
(`reeds_shepp_path`, `reeds_shepp_path_pose`, `reeds_shepp_length`).

The cusp filter rejects reference paths whose reverse folds are too short or too
close to a forward leg. Defaults from `config/curriculum.yaml`:

| Key | Default | Effect |
| --- | ---: | --- |
| `min_cusp_fold_arc` | 0.8 m | reject a maximal reverse segment shorter than this |
| `min_cusp_leg_separation` | 0.35 m | reject a reverse leg closer than this to a forward leg |
| `cusp_min_arc_gap` | 1.0 m | only compare leg pairs separated by at least this arc length |

Set a threshold to `0` to disable that criterion. A rejected candidate is
resampled; `MapGenerator.cusp_rejections` counts the rejections.

Without numba, core scenarios still run; planning scenarios raise an `ImportError`
whose message contains the install hint:

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

Output:

```text
planning imports: MapGenerator hybrid_astar_search reeds_shepp_length
generated map: obstacles=5 path=(152, 3) factory.stats={'attempts': 1, 'retries': 0, 'failures': 0, 'episodes': 1}
ImportError: 场景 'obstacles_sparse' 需要 Hybrid-A* 地图生成器（chassis_sim.planning），它依赖 numba。请先安装 planning extra：pip install chassis-sim[planning]
```

## Rendering

`chassis_sim.render` draws the current bridge state with matplotlib using the
`Agg` backend, so it works headless. `render_to_png()` writes one frame;
`render_episode_frames()` steps the bridge through an action sequence and writes
frames (always including the last step); `save_gif()` assembles frames. matplotlib
is imported lazily and is only needed for rendering. `imageio` is optional: if it
is missing, `save_gif()` writes a directory of frames and emits a warning instead
of raising.

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
print("png:", png, f"{png.stat().st_size} bytes")

actions = [
    np.array([0.2 * np.sin(index / 5.0), 0.4], dtype=np.float64)
    for index in range(10)
]
frames = render_episode_frames(env.bridge, actions, out_dir, every=5, prefix="step", dpi=80)
print("frames:", [frame.name for frame in frames])
print("sizes:", [frame.stat().st_size for frame in frames])
```

Output:

```text
png: /tmp/chassis_sim_readme_render/frame.png 2812 bytes
frames: ['step_00005.png', 'step_00010.png']
sizes: [3133, 3284]
```

## Examples

| Script | What it does |
| --- | --- |
| `examples/dynamic_demo.py` | static vs dynamic `dynamic_crossing` over 20 steps; checks bit-identical replay without dynamic obstacles |
| `examples/eval_matrix.py` | sparse smoke run: 3 controllers × 3 chassis × 10 independent maps at `max_steps=300`; writes Markdown and JSON |
| `examples/render_demo.py` | 50 steps, 5 PNG frames, and a seven-round render timing |
| `examples/scenario_demo.py` | `straight`, `corridor`, and `dense_static` for 50 random steps each |
| `examples/sensors_demo.py` | IMU, wheel odometry, and parametric LiDAR with latency and dropout |

Run them from the repository root with the same interpreter that has the package
installed.

## Known limitations

1. **Discrete collision test, no swept body.** Collision is a discrete per-step test
   of the current body OBB against circles and oriented rectangles (with a safety
   margin); there is no swept-body / continuous (CCD) check, so a fast reverse can
   still clip a corner between ticks.
2. **DWA gear selection is off by default.** `gear_mode="auto"` truncates
   candidates to one direction and is sensitive to `heading_lookahead`; with the
   discrete collision test above, a hard gear choice can collide during a maneuver
   that a full-range candidate set would avoid.
3. **`get_scan()` metadata is historical.** `get_scan()` returns the ten classic
   LaserScan keys with the fixed constants 180 rays, π FOV, 10 m maximum range, and
   0.05 m minimum range. Parameterized LiDAR settings (rays, FOV, range, latency,
   dropout) are reported only in `info["scan_meta"]`.
4. **No camera.** There is no RGB, depth, or segmentation sensor.
5. **Sensors do not enter the observation.** IMU, wheel odometry, and LiDAR readings
   are exposed through `info` only. An observation encoder that uses them must be
   built outside the environment.
6. **One map per core scenario.** Core scenarios are fixed YAML maps; only the
   planning scenarios (and a custom callable) produce a new map per episode.

## License, contributing, and citation

MIT. See `LICENSE`.

Tests and lint:

```bash
pytest -q
ruff check .
```

CI runs the test suite on Python 3.10, 3.11, and 3.12
(`.github/workflows/ci.yml`).

If you use this package in academic work, cite the repository:

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

All numbers in this README were measured on 2026-11-03 (UTC) with Python 3.12.0,
NumPy 1.26.4, and Gymnasium 1.3.0. The evaluation matrix uses ten independent maps,
`max_steps=500`, and `base_seed=0`. The code blocks in this file are executed as
written in the repository's verification pass.
