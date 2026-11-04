# Changelog

All notable changes to `chassis-sim`. This file documents the changes present in
the working tree; the most recent release is `3.0.0`.

## 3.0.0

3.0.0 是 API 稳定线：从这一版起本包承诺 semver，因此 classifier 保持
`Development Status :: 5 - Production/Stable`，不降级为 Beta。

### Breaking

- **Gymnasium id 由 `-v0` 改成 `-v1`**：`ChassisSim-{Ackermann,DiffDrive,Mecanum}-v0` 不再注册，
  `gym.make("ChassisSim-Ackermann-v0")` 抛 `gym.error.DeprecatedEnv`，消息里直接点名要用的 `-v1` id。
  三个 `-v1` id 的 `entry_point`、动作空间与 `max_episode_steps` 与 2.x 相同。
- **默认场景改为 `dense_static`**：`EnvConfig.default_scenario` 默认 `"dense_static"`，`reset()` 不带
  显式 seed 时每回合重掷地图（`default_scenario=None` 才回到 2.x 的 `straight` 直线兜底）。
  2.x 里 `reset()` 拿到的是固定直线场景，所以这既是默认值变更，也会改变无 seed 训练的分布。
- **`observation_space` 从无界 `Box(±inf)` 变成全有限**：各段按物理上界声明——`path` 段的 ρ 上界是
  `world_diag + lookahead + 0.1 = 34.24213790893555`、角度段 π、归一化段 `1 + 1 ulp = 1.0000001192092896`；
  非零有限界各放宽 1 ulp。真正越界的观测值才被 clip，并在 `info["obs_clipped"]` 给出计数（计数为 0 时该键不出现）。
  依赖 `low`/`high` 为 `inf` 的包装器要改用新值。
- **闭式积分替换 Euler 步进**：同一动作序列下位姿会变——圆周 max-radial 误差 `5.04e-2` → `3.33e-16` m，
  60 步开环序列的 xy 最大差 `0.0403` m，θ 通道逐位不变；`|ω·dt| ≤ 1.49e-8`（`sqrt(float64 eps)`）时仍走
  与 2.x **逐位相同**的 Euler 分支。用 2.x 生成的 golden 必须重录：本仓 `tests/data/ackermann_golden.json`
  已重录，旧轨迹以 `poses_euler_legacy` 保留并与 HEAD 的 poses `np.array_equal=True`。

### Added

- **观测段可声明物理上界**：新增 `BoundContext`（`dt` / `v_max` / `v_min` / `vy_max` / `omega_max` /
  `lookahead` / `world_diag` / `lidar`）与 `ResolvedObsSpec.bounds()`，内置段各自给出界，
  `Custom` 段也能带自己的界；`observation_space` 由这些界拼成，不再是无界 Box。
- **渲染契约**：`render_mode` 支持 `None` / `"human"` / `"rgb_array"`，`render()` 返回 `uint8 (600, 600, 3)`，
  `metadata` 带 `render_modes` 与 `render_fps = 10`；新增 `chassis_sim.render.render_to_array(bridge, dpi=...)`
  供无窗口取帧。缺 `matplotlib` 时抛带 `pip install "chassis-sim[viz]"` 的 `ImportError`。
- **轮级运动学层（纯只读）**：`Chassis.wheel_kinematics(cmd) -> WheelKinematics | None`，给出 Ackermann 的
  内外前轮角、DiffDrive 的左右轮速、Mecanum 的四轮速（逆 Jacobian）；Ackermann 另有 `front_wheel_angles(delta)`。
  不支持时返回 `None`，不返回空 dict；积分器与动作语义一律未动。
- **`integration_substeps` 与 `ccd`**：前者把一步切成 N 段分别积分与碰撞采样，后者在每段内按 `0.02 m` /
  `0.05 rad` 采样内部位姿（上限 64 个）以拦住薄墙穿透。默认 `1` / `False`，与 2.x 逐位相同；
  `ccd` 只接受 `bool`，`1` / `0` / `"false"` 一律 `ValueError`。
- **基准协议**：`chassis_sim.eval.benchmark(...)`（策略 × held-out 场景 × ≥3 seed、同 (场景, seed) 的地图在
  策略间共享做配对、Wilson 区间、独立样本闸门——地图指纹有重复时 `success_ci` 返回 `None` 而不是给假数字），
  四基线 `zero` / `random` / `pp` / `dwa`，以及 `python -m chassis_sim benchmark` 子命令。`evaluate()` 签名未动。
- **`EnvConfig.default_scenario`**：默认 `"dense_static"`，设为 `None` 回到 2.x 的直线兜底。
- **CI 的 ubuntu 矩阵覆盖 NumPy 1.x 与 2.x 两档**（`matrix.numpy`：`numpy<2` /
  `numpy>=2`），每档都跑完整套件，并用一步断言实测到的 NumPy 主版本与矩阵档位一致。
  `3.13 × numpy<2` 显式排除：PyPI 上 cp313 的 NumPy 从 2.1.0 才开始发布
  （`pip download --python-version 3.13 "numpy<2"` 报
  `No matching distribution found`），pip 只能退回源码构建。3.13 仍在
  `numpy>=2` 档真跑：numba 的 cp313 轮子从 0.61.0 起提供，`numba>=0.60` 在
  3.13 上会解析到 0.61+。
- **新增 CI 的 `docs` job 与 `tools/check_readme_blocks.py`**：把两份 README 里的
  `python` 围栏块逐个放进独立子进程执行，任一失败即红；块首行写 `# no-exec`
  才允许跳过，跳过计入 skipped 而不是 passed，一个 python 块都没有也算失败。
- **`dev` extra 自带全部闸门工具**：`mypy==2.4.1`、`pytest-cov==7.1.0`、
  `types-PyYAML==6.0.12.20261103`。`pip install -e ".[dev]"` 之后即可直接跑
  `MYPYPATH=src mypy` 与 `pytest --cov`，不必再单独装工具。

### Changed

- **planning extra 的 numba 下界 `>=0.56` → `>=0.60`**：Numba 0.60.0 是第一个提供
  NumPy 2.0 二进制支持的版本，更早的组合与 `numpy>=2` 在 import 阶段就失败。
  运行依赖 `numpy>=1.24` 不变 —— 1.x 与 2.x 都仍然支持。
- **README 与包定位改口径**：从 “Gymnasium benchmark” 改为「底盘控制 + 观测/奖励工作台」，并写明三条边界
  （底盘级运动学而非轮级刚体动力学、碰撞是几何判据无接触力、无 GPU 并行）；`CITATION.cff` 的标题与摘要同步。
- **闭式积分下的 `integration_substeps` 语义**：精度不再随分段数改善（已到浮点噪声），它的作用只剩提高碰撞采样密度；
  文档与 docstring 都按这个口径写，不宣称「更高精度积分」。

### Fixed

- **注入的桥不再被塞进 env 的内置场景**：2.x 起 `ChassisSimEnv(bridge=...)` 且 `reset()` 不带 `options` 时，
  env 会把 `default_scenario` 的 `obstacles` / `path_waypoints` / `initial_heading` 直接递给**调用方的桥**，
  而桥照单全收、不报错（主仓的 `PhysicalCommandBridge(NumpySimBridge)` 正是这种注入形态）。现在只有
  「`options is None` + 桥由 env 自造 + 无 `map_generator` + 配了 `default_scenario`」四条全成立才接管；
  注入桥保留调用方自己的世界。
- **`render()` 与渲染元数据补齐**：2.x 的 `metadata` 只有 `render_modes=["human"]`、没有 `render_fps`，
  `env.render()` 直接 `NotImplementedError`，`gym.make(..., render_mode="rgb_array")` 会 `TypeError`。
  3.0 起三者都按 Gymnasium 契约工作，`gymnasium.utils.env_checker.check_env(..., skip_render_check=False)`
  在三种底盘上都不再报 `Box maximum value is infinity`。

## 2.0.0

### Breaking

- **`ChassisSimEnv(device=...)` and `cs.make(..., device=...)` are gone.** The parameter
  had been inert since 1.0 — the encoder lives in the Actor and `ContextBuilder` only
  computes numpy — and was documented as inert. A no-op keyword is still public API, so
  removing it is a major bump rather than a patch: `ChassisSimEnv(device="cpu")` now
  raises `TypeError: __init__() got an unexpected keyword argument 'device'`, and
  `cs.make(device="cpu")` raises `ValueError: 未知的 make() 参数 ['device']` followed by
  the 57 accepted keys. Grep training scripts for `device=` before upgrading; nothing
  else about the constructor changed.
- **The matching entry is gone from both READMEs' `Known limitations` sections**, which
  are renumbered (nine items, no gap).

### Changed

- **CI now runs the same suite on macOS and Windows** (new `test-os` job: Python 3.12,
  `[dev,viz,planning]`, `shell: bash`, `PYTHONUTF8=1`), so the declared
  `Operating System :: OS Independent` classifier has something behind it instead of an
  ubuntu-only history. ubuntu still covers 3.10–3.13. The matrix values are written as
  arrays: a scalar `python-version` keeps the YAML valid but makes GitHub reject the whole
  workflow (the run then fails with zero jobs and no logs), so `tests/test_ci_workflow.py`
  pins that shape as well as the job's `shell: bash` steps.
- **The planning parity golden compares `cusp_rejections` only in strict mode.** That
  count is discrete: both of its criteria (`min_cusp_fold_arc`, `min_cusp_leg_separation`)
  compare a float against a fixed threshold, so a single ulp decides whether one candidate
  is rejected. On the first macOS/Windows run the generated obstacles, path and headings
  matched inside `1e-9` while the counter read 1 vs 0 and 1 vs 2 — the same path after a
  different number of retries, not a porting error. `CHASSIS_STRICT_GOLDEN=1` still
  asserts it, and the invariant that matters — a generated path contains no violating
  cusp — is asserted independently by `test_cusp_filter_effective_on_generated_paths`.

### Fixed

- **Worker crash logs no longer vanish on Windows.** `reset()` and `step()` wrote the
  traceback to a hard-coded `/tmp/ackermann_worker_crash.log`; Windows has no `/tmp`, so
  `open()` raised `FileNotFoundError`, the surrounding `except` swallowed it, and the
  diagnostic disappeared. The path now comes from `tempfile.gettempdir()` and the write is
  explicitly UTF-8 — a Chinese traceback cannot be encoded through the default ANSI code
  page either.

No other functional changes: observation layout, rewards, sensors, controllers, planning
and rendering are untouched in this release.

## 1.1.1

### Changed

- **The gallery images moved from `docs/media/` to `assets/`, and `docs/` is no longer
  tracked.** The wiki-style documents under `docs/` were internal working notes; they are
  not part of the library and are no longer published. No code, no public API and no
  observation/sensor behaviour changed in this release.
- **Both READMEs now describe only the current release.** The migration notes, the
  "0.1.x line vs 1.x line" comparisons and the "since 1.1.0" phrasing are gone; every
  statement is present-tense so the text does not rot with each release. Version numbers
  live in `pyproject.toml` and in this changelog, not in the README prose.

## 1.1.0

### Added

- **`EnvConfig.seed` — a run seed that actually reproduces the run.** With `domain_rand=True`
  the LiDAR noise and the odometry drift used to come from the global `np.random`, so two
  runs with the same seed diverged. `SimpleBridge(seed=...)` now owns a
  `np.random.Generator` and every domain-randomisation draw goes through it;
  `cs.make(seed=...)` sets the run seed and — when `sensor_seed` is not given — also the
  sensor streams, through one rule implemented once
  (`chassis_sim.config.resolve_sensor_seed`):
  `sensor_seed if sensor_seed is not None else (seed if seed is not None else 0)`.
  A bridge you construct yourself keeps ownership of its own randomness.
  Bit-for-bit compatibility: with `seed=None` the bridge still draws from the global
  `np.random` (verified identical to 1.0.2 on a clean worktree), and `make(seed=3)` alone
  still yields the same sensor streams and scenario maps as 1.0.2.

### Changed

- **`env.reset()` now consumes `env.default_reset_options`.** `cs.make(scenario=...)` already
  stored the scenario keys on the env, but `reset()` ignored them unless you handed them
  over yourself — a forgotten `options=` silently fell back to the empty field. From this
  release `reset(options=None)` falls back to the stored defaults, while an explicit
  `options=` (including `{}`) always wins.

No other functional changes.

## 1.0.2

### Changed

- **The metadata now describes what the package is.** Classifiers say
  `Development Status :: 5 - Production/Stable` instead of `4 - Beta`, and gained
  `Programming Language :: Python :: 3.13` and `Typing :: Typed`. The CI test matrix
  grew a 3.13 entry in the same commit, so the new claim is backed by an actual run
  rather than assertion.
- **The docs no longer hard-code the release they describe.** Sentences that described the
  current product now say `1.x`, and the install example uses `==X.Y.Z` instead of a pinned
  literal: `pyproject.toml` is the single source of truth, so no version number rots inside
  prose on every release.

### Added

- `CITATION.cff` — gives GitHub's "Cite this repository" button for academic users.
- `CONTRIBUTING.md` — the five gates with their exact commands, checked line-by-line against
  `.github/workflows/ci.yml`, plus the repository conventions (Chinese comments and error
  messages, no lowering of the coverage floor, both READMEs kept in sync).
- `SECURITY.md` — supported range, private reporting through GitHub Security Advisories,
  response targets, and what is explicitly out of scope.
- `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.yml` — issue forms that ask for the
  version, Python, OS and a minimal reproduction.
- `.pre-commit-config.yaml` — ruff 0.17.3 plus pre-commit-hooks.

No functional changes: the package code is identical to 1.0.1.

## 1.0.1

### Fixed

- **Project links follow the GitHub rename** (`zhao6663` → `ravenkernel`). The badges, the
  eleven gallery image URLs, the release/issue links and `[project.urls]` now point at
  `github.com/ravenkernel/chassis-sim`. The old links still resolve through GitHub's
  redirect, but a redirect is not a contract.
- **The metadata published as 1.0.0 could not be corrected in place.** Project URLs and the
  long description are immutable once uploaded, so 1.0.1 re-publishes the same content with
  the new owner (the images inside the 1.0.0 PyPI description pointed at the old account).

No functional changes: the package code is identical to 1.0.0.

## 1.0.0

The first release with one public surface. Everything a user is meant to touch is
reachable as `chassis_sim.<name>`: `make`, `EnvConfig`, the observation segments,
the sensors, the controllers, `evaluate`, `record`, `snapshot`. Deep imports keep
working, but they are no longer the documented path.

### Breaking

- **The default observation is 128 values (129 on the mecanum), and there is no
  built-in 494 preset.** `ObsSpec.default(action_dim, points=30)` is
  `odom + path + goal + act` = 4 + 120 + 2 + action_dim. A 494-value vector now
  requires an explicit nine-segment `ObsSpec` (lidar / odom / path / goal / act /
  err / goal_sig / path_valid / path_progress) plus the sensors the `LidarSegment`
  names.
- **`ObsLayout`, `ObservationBuilder`, `compute_obs_dims` and
  `BASE_RAW_OBS_DIM` are deleted.** No alias, no deprecation shim: the observation
  layout is the `ObsSpec` you pass, and `obs_spec_describe()` is the only place
  that publishes slices.
- **`chassis_sim.obs.Path` is now `PathSegment`.** The rename avoids reading like
  the `chassis_sim.paths` module.
- **The per-sensor LiDAR accessors are gone.** `get_scan()`, `get_lidar_front()`,
  `get_lidar_rear()` and `scan_to_pointcloud()` are deleted;
  `bridge.get_sensor_reading()` is the single entry point and returns one dict keyed
  by sensor name. Sensor readings reach the environment as `info[<sensor name>]`,
  not `info["scan"]` / `info["imu"]` / `info["wheel"]`.
- **The deep paths under `chassis_sim.envs` moved to stable modules.**
  `chassis_sim.paths` (PathExtractor, cartesian_to_polar, np_angle_wrap,
  segment_is_reverse, gear_is_compliant, GEAR_EPS), `chassis_sim.safety`
  (SafetyGate), `chassis_sim.reward` (RewardCalculator), `chassis_sim.config`
  (EnvConfig, RewardConfig, from_config_dir, load_default_config).
  `chassis_sim.envs` now exports exactly one name: `ChassisSimEnv`.
- **`evaluate(n_episodes=)` is `evaluate(episodes=)`.** The old keyword is gone,
  not aliased.
- **`EnvConfig.domain_rand_enabled` defaults to `False`.** It was `True`; the
  library is now deterministic unless domain randomization is asked for. The
  three-layer resolution is `make(domain_rand=...) > cfg.domain_rand_enabled >
  False`.
- **A default environment no longer assembles a LiDAR.** `cs.make("ackermann")`
  carries no sensor, so `info["d_min"]` equals `bridge.max_range` (10.0 m by
  default) and the reward's obstacle term degenerates to a constant. A radar has to
  be declared with `sensors=[cs.Lidar(...)]`.

### Added

- `cs.make(chassis_or_cfg, **kwargs)` — one constructor for a name or an
  `EnvConfig`, resolving **kwargs > cfg field > library default** per field, with
  `obs=`/`obs_spec=` and `seed=`/`sensor_seed=` aliases and an explicit
  `ValueError` for unknown keys.
- `EnvConfig.to_yaml(path)` / `EnvConfig.from_yaml(path)` — a full 44-field
  snapshot, so a run is reproducible from one file.
- `cs.make_controller(kind, env=...)` (or `chassis=...`) — the documented way to
  build a controller for a chassis, instead of instantiating a controller class.
- `evaluate(env, episodes=...)` with the controller omitted — the evaluator binds
  pure pursuit to the environment's own chassis.
- `EvalResult.markdown()` (`to_markdown()` alias) — the two Markdown tables the
  README quotes, including the fixed-map warning.
- `cs.record(env, controller, out=..., every=..., fps=...)` and
  `cs.snapshot(env, path)` — one call for a GIF of an episode and one for a still.
  The still helper is named `snapshot` because `chassis_sim.render` is the drawing
  subpackage: importing a submodule rebinds the parent attribute of the same name to
  the module, so a function called `render` cannot survive at the top level.
- `chassis-sim` console script and `python -m chassis_sim`, with
  `list-scenarios`, `record`, `eval` and `render`. `record` and `render` exit
  with status 2 and a one-line install hint when `matplotlib` / `imageio` is
  missing.
- `chassis_sim.protocols` — the `Bridge` protocol and friends, importable without
  pulling matplotlib, imageio or numba.
- `chassis_sim.paths`, `chassis_sim.safety`, `chassis_sim.reward` as stable,
  documented modules rather than private corners of `chassis_sim.envs`.
- Quality gates wired into the repo: `ruff check` + `ruff format --check`, a
  `mypy` file list, and a coverage floor, all configured in `pyproject.toml` and
  run by CI.

### Removed

Listed again so a grep over an old codebase finds them:

- `ObsLayout`, `ObservationBuilder`, `compute_obs_dims`, `BASE_RAW_OBS_DIM`
- `chassis_sim.obs.Path` (use `PathSegment`)
- `get_scan()`, `get_lidar_front()`, `get_lidar_rear()`, `scan_to_pointcloud()`
  (use `get_sensor_reading()`)
- `chassis_sim.envs.path_extractor`, `chassis_sim.envs.path_utils`,
  `chassis_sim.envs.safety_gate`, `chassis_sim.envs.reward`,
  `chassis_sim.envs.config`, `chassis_sim.envs.observation` (use
  `chassis_sim.paths` / `safety` / `reward` / `config`)
- the other `chassis_sim.envs.*` re-exports (only `ChassisSimEnv` remains)
- `evaluate(..., n_episodes=)`

### Fixed

- **`EnvConfig.emit_reward_components` was silently overridden.** The constructor
  default of `ChassisSimEnv.emit_reward_components` won over the config field, so a
  config that asked for reward components either lost them or emitted them
  unexpectedly. The field now takes effect, and the resolution is `kwargs > cfg`
  everywhere.
- **The `get_scan()` metadata mismatch disappeared with the interface.** The removed
  helper reported the ten classic LaserScan keys with fixed constants (180 rays,
  `π` FOV, 10 m maximum, 0.05 m minimum) no matter how the radar was actually
  parameterized; `info[<sensor name>]["scan_meta"]` is parameterized by construction.

## 0.2.1

### Fixed

- **The install instructions now match reality.** `0.2.0` shipped with a README that still
  said the package was not on PyPI (written before the upload). `pip install chassis-sim` is
  the supported entry point; the wheel attached to each GitHub release stays available for
  pinning to a tag.
- **The language switch works from the PyPI page too.** Both READMEs link to each other
  with absolute `blob/main` URLs instead of repo-relative paths, which would 404 on
  pypi.org.

No functional changes: the simulator, the observation segments and the sensor specs are
identical to `0.2.0`.

## 0.2.0

### Breaking

- **The default observation is no longer 494 values.** `ObsSpec.default(action_dim, path_points=30)`
  builds `odom + path + goal + act`, which is **128** values on the Ackermann and
  differential-drive chassis and **129** on the mecanum (the extra value is the
  third action component). The old 494-value layout is now an explicit choice:
  `ObsSpec` assembled from `ObsLayout(2, 30).as_spec()`, passed as
  `ChassisSimEnv(..., obs_spec=...)`.
- **`SensorConfig` is gone; sensors are a list of specs.** Build
  `Lidar` / `Imu` / `WheelOdom` instances and pass them as
  `ChassisSimEnv(..., sensors=[...], sensor_seed=0)` or
  `SimpleBridge(chassis, sensors=[...], sensor_seed=0)`. `sensors=` and an explicit
  `bridge=` are mutually exclusive.
- **`chassis_sim/bridges/sensors.py` was deleted.** The old single-config sensor
  bridge no longer exists; sensor state lives in `chassis_sim.sensors` and in the
  bridge's own `get_sensor_reading()`.
- Sensor readings stay in `info` (`info["scan"]`, `info["imu"]`, `info["wheel"]`)
  and never enter the observation vector unless a `LidarSegment` observation segment
  explicitly names the sensor. Reset now carries the same readings as `step()`
  (the bridge produces a real first frame inside `reset()`), and still returns
  `{}` when no sensor is configured.
- **`chassis_sim.obs.Lidar` → `LidarSegment`**（与 `chassis_sim.sensors.Lidar` 重名）.
  The observation segment is now `chassis_sim.obs.LidarSegment`; the sensor spec
  keeps the name `chassis_sim.sensors.Lidar` and there is no compatibility alias.

### Added

- `chassis_sim.obs`: composable `ObsSpec` with segments `Odom`, `Path`,
  `Goal`, `Act`, `Err`, `GoalSig`, `PathValid`, `PathProgress`, `LidarSegment`, and
  `Custom`; plus `ObsSpec.default()`, `ObsSpec.resolved()`,
  `ResolvedObsSpec.total_dim`, `ResolvedObsSpec.describe()`, and
  `env.obs_spec_describe()` returning `name -> slice`.
- `chassis_sim.sensors`: list-style specs `Lidar(name, mount, fov, n_rays,
  max_range, range_min, noise_std, quant_step, latency_steps, dropout_prob)`,
  `Imu(name, noise_std)`, `WheelOdom(name, track_width, noise_std)`. Each sensor
  owns a random stream derived from `(sensor_seed, name)`, so adding or removing a
  sensor never shifts the noise of the others.
- Missing-frame contract: the bridge produces a real first LiDAR frame right after
  `reset()`; a delayed or dropped frame makes the `Lidar` observation segment repeat
  the last valid scan (zero-order hold), and `max_range` is used only before the
  first valid frame. `env.reset()` clears the hold cache. `scan_meta["valid"]` /
  `scan_meta["dropped"]` remain the only validity signal.
- `examples/gallery.py`, which regenerates the five clips and six scenario stills
  used by the README gallery, and the README gallery section itself.
- README rewritten for 0.2.0: the observation contract, the explicit 494 preset,
  the sensor spec list, the zero-order hold contract, a regenerated evaluation
  matrix, and executable, output-checked code blocks throughout.

### Fixed

- The README documented `PhysicalCommandBridge.step()` as accepting physical
  commands. It accepts the **normalized** action and calls
  `inner.step(chassis.physical_action(action))`; `to_physical()` is the pure
  mapping. Documentation corrected, behavior unchanged.
- Documentation no longer refers to `SensorConfig`, `LidarParams`,
  `bridges/sensors.py`, the `side=` argument, or `get_lidar_front`, and no longer
  describes a fixed 494-value observation outside the explicit preset section.
