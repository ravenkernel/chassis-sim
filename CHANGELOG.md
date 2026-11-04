# Changelog

All notable changes to `chassis-sim`. This file documents the changes present in
the working tree; the most recent release is `1.1.0`.

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
