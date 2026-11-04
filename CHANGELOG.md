# Changelog

All notable changes to `chassis-sim`. This file documents the changes present in
the working tree; the most recent published tag is `v0.2.1`.

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
