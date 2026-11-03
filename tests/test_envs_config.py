"""配置层契约：load_default_config 与 CWD 无关、包内 YAML 是 package data、
from_config_dir 的校验与 train.yaml 可选语义。

背景（主仓 bug）：training/config.py:976-1001 的 from_config_dir 用
`base = Path(config_dir)` + `open(base / "train.yaml")` + OmegaConf.load，
装成 wheel 后 CWD 不在仓库根就找不到文件。本包改为 importlib.resources，
并且不再发布 train.yaml（它是训练配置，不属于 env 层）。
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import pytest
import yaml

import chassis_sim.config as pkg
from chassis_sim.config import (
    EnvConfig,
    RewardConfig,
    from_config_dir,
    load_default_config,
)

PKG_DIR = Path(pkg.__file__).resolve().parent


def _write_minimal(
    tmp_path: Path, reward_params: dict | None = None, weights: dict | None = None
) -> Path:
    """写一份最小可用的 reward.yaml + env.yaml，返回目录。"""
    reward = {
        "reward_weights": {"cte_penalty": 6.0} if weights is None else weights,
        "reward_params": {} if reward_params is None else reward_params,
    }
    (tmp_path / "reward.yaml").write_text(
        yaml.safe_dump(reward, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "env.yaml").write_text(
        "world:\n  step_time: 0.1\nrobot:\n  shape:\n    wheelbase: 0.145\n",
        encoding="utf-8",
    )
    return tmp_path


class TestLoadDefaultConfig:
    """包内 YAML + importlib.resources ⇒ 不依赖 CWD。"""

    def test_works_from_unrelated_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # 证明不是靠 CWD 找到 config/
        env, rew = load_default_config()
        assert isinstance(env, EnvConfig)
        assert isinstance(rew, RewardConfig)

    def test_equals_dataclass_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env, rew = load_default_config()
        assert env == EnvConfig()
        assert rew == RewardConfig()

    def test_env_config_key_values(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env, _ = load_default_config()
        assert env.dt == pytest.approx(0.1)
        assert env.wheelbase == pytest.approx(0.145)
        assert env.lidar_max_range == pytest.approx(10.0)
        assert env.v_min == pytest.approx(-0.6)
        assert env.v_max == pytest.approx(1.2)
        assert env.delta_max == pytest.approx(0.526)
        assert env.obs_danger == pytest.approx(0.3)
        assert env.obs_safe == pytest.approx(1.0)
        assert env.stop_threshold == pytest.approx(0.2)
        assert env.privileged_enabled is False
        assert env.privileged_bins == 36

    def test_reward_config_key_values(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _, rew = load_default_config()
        assert rew.obstacle_range == pytest.approx(0.8)
        assert rew.goal_bonus == pytest.approx(400.0)
        assert rew.collision_penalty == pytest.approx(-100.0)
        assert rew.collision_surcharge == pytest.approx(60.0)
        assert rew.step_penalty == pytest.approx(-0.02)
        assert rew.failure_dense_cap == pytest.approx(0.12)
        assert rew.failure_dense_refund_cap == pytest.approx(120.0)
        assert rew.stage_scale_floor == pytest.approx(0.2)
        assert rew.reverse_motion_penalty == pytest.approx(2.5)
        assert rew.obstacle_speed_power == pytest.approx(0.0)
        assert rew.goal_time_bonus == pytest.approx(0.0)
        assert rew.path_d_max == pytest.approx(4.0)

    def test_reward_weights_match_dataclass(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _, rew = load_default_config()
        assert set(rew.reward_weights) == set(RewardConfig().reward_weights)
        for key, want in RewardConfig().reward_weights.items():
            assert float(rew.reward_weights[key]) == pytest.approx(float(want)), key

    def test_repeated_calls_are_stable(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert load_default_config() == load_default_config()


class TestPackagedYaml:
    """三个 YAML 必须真的随 wheel 发布，且顶层键与主仓一致。"""

    @pytest.mark.parametrize("name", ["env.yaml", "reward.yaml", "curriculum.yaml"])
    def test_package_data_is_readable(self, name):
        res = importlib.resources.files("chassis_sim.config").joinpath(name)
        assert res.is_file(), f"{name} 未作为 package data 打进 wheel"
        assert yaml.safe_load(res.read_text(encoding="utf-8")) is not None

    def test_env_yaml_top_level_keys(self):
        raw = yaml.safe_load(
            importlib.resources.files("chassis_sim.config")
            .joinpath("env.yaml")
            .read_text(encoding="utf-8")
        )
        assert set(raw) == {"world", "robot"}
        assert raw["robot"]["vel_dim"] == 2
        assert raw["robot"]["shape"]["wheelbase"] == pytest.approx(0.145)
        sensors = raw["robot"]["sensors"]
        assert [s["type"] for s in sensors] == ["lidar2d", "lidar2d"]
        assert all(s["number"] == 180 for s in sensors)

    def test_reward_yaml_top_level_keys(self):
        raw = yaml.safe_load(
            importlib.resources.files("chassis_sim.config")
            .joinpath("reward.yaml")
            .read_text(encoding="utf-8")
        )
        assert set(raw) == {"reward_weights", "reward_params"}
        assert raw["reward_params"]["failure_dense_cap"] == pytest.approx(0.12)
        assert raw["reward_params"]["failure_dense_refund_cap"] == pytest.approx(120.0)

    def test_curriculum_yaml_top_level_keys(self):
        raw = yaml.safe_load(
            importlib.resources.files("chassis_sim.config")
            .joinpath("curriculum.yaml")
            .read_text(encoding="utf-8")
        )
        assert set(raw) == {"active_stage", "defaults", "stages"}

    def test_packaged_dir_is_loadable_directly(self):
        env, rew = from_config_dir(PKG_DIR)
        assert env == EnvConfig()
        assert rew == RewardConfig()


class TestFromConfigDir:
    """用户目录读法、必需文件、错误路径。"""

    def test_reads_user_directory(self, tmp_path):
        _write_minimal(
            tmp_path, reward_params={"obstacle_range": 1.25, "goal_bonus": 250.0}
        )
        env, rew = from_config_dir(tmp_path)
        assert rew.obstacle_range == pytest.approx(1.25)
        assert rew.goal_bonus == pytest.approx(250.0)
        assert env.dt == pytest.approx(0.1)

    def test_accepts_str_path(self, tmp_path):
        _write_minimal(tmp_path)
        env, rew = from_config_dir(str(tmp_path))
        assert isinstance(env, EnvConfig) and isinstance(rew, RewardConfig)

    def test_missing_reward_yaml_raises(self, tmp_path):
        (tmp_path / "env.yaml").write_text("world: {}\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            from_config_dir(tmp_path)

    def test_missing_env_yaml_raises(self, tmp_path):
        _write_minimal(tmp_path)
        (tmp_path / "env.yaml").unlink()
        with pytest.raises(FileNotFoundError):
            from_config_dir(tmp_path)

    def test_non_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            from_config_dir(tmp_path / "does-not-exist")

    def test_train_yaml_is_optional(self, tmp_path):
        _write_minimal(tmp_path)
        env, _ = from_config_dir(tmp_path)
        assert env == EnvConfig()  # 全部落回字段默认值

    def test_train_yaml_is_honored_when_present(self, tmp_path):
        _write_minimal(tmp_path)
        (tmp_path / "train.yaml").write_text(
            "env:\n"
            "  v_max: 1.7\n"
            "  delta_max: 0.6\n"
            "privileged:\n"
            "  enabled: true\n"
            "  n_bins: 24\n"
            "display:\n"
            "  envs_panel: false\n",
            encoding="utf-8",
        )
        env, _ = from_config_dir(tmp_path)
        assert env.v_max == pytest.approx(1.7)
        assert env.delta_max == pytest.approx(0.6)
        assert env.privileged_enabled is True
        assert env.privileged_bins == 24
        assert env.envs_panel_enabled is False

    def test_env_yaml_supplies_world_and_robot(self, tmp_path):
        """dt / wheelbase / lidar_max_range 来自 **env.yaml**（与主仓取值顺序一致）。"""
        _write_minimal(tmp_path)
        (tmp_path / "env.yaml").write_text(
            "world:\n  step_time: 0.05\n"
            "robot:\n  shape:\n    wheelbase: 0.2\n"
            "  sensors:\n    - type: lidar2d\n      range_max: 8.0\n"
            "    - type: lidar2d\n      range_max: 99.0\n",
            encoding="utf-8",
        )
        env, _ = from_config_dir(tmp_path)
        assert env.dt == pytest.approx(0.05)
        assert env.wheelbase == pytest.approx(0.2)
        assert env.lidar_max_range == pytest.approx(8.0)  # 取**第一个** lidar2d

    def test_obs_thresholds_come_from_reward_yaml(self, tmp_path):
        _write_minimal(
            tmp_path,
            reward_params={"obs_danger": 0.25, "obs_safe": 1.4, "stop_threshold": 0.15},
        )
        env, _ = from_config_dir(tmp_path)
        assert env.obs_danger == pytest.approx(0.25)
        assert env.obs_safe == pytest.approx(1.4)
        assert env.stop_threshold == pytest.approx(0.15)


class TestValidation:
    """未知键 / 非法取值必须在加载期拒绝（与 OmegaConf structured 的拒绝面对齐）。"""

    def test_unknown_reward_params_key_rejected(self, tmp_path):
        _write_minimal(tmp_path, reward_params={"bogus_key": 1.0})
        with pytest.raises(ValueError, match="未知键"):
            from_config_dir(tmp_path)

    def test_unknown_reward_weight_rejected(self, tmp_path):
        _write_minimal(tmp_path, weights={"cte_penalty": 6.0, "not_a_weight": 1.0})
        with pytest.raises(ValueError, match="未知键"):
            from_config_dir(tmp_path)

    def test_unknown_top_level_key_rejected(self, tmp_path):
        _write_minimal(tmp_path)
        reward_path = tmp_path / "reward.yaml"
        raw = yaml.safe_load(reward_path.read_text(encoding="utf-8"))
        raw["reward_extra"] = {}
        reward_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="未知键"):
            from_config_dir(tmp_path)

    def test_wrong_scalar_type_rejected(self, tmp_path):
        _write_minimal(tmp_path, reward_params={"obstacle_range": "0.8"})
        with pytest.raises(ValueError, match="类型必须是 float"):
            from_config_dir(tmp_path)

    @pytest.mark.parametrize(
        "params, needle",
        [
            ({"collision_surcharge": -1.0}, "collision_surcharge"),
            ({"timeout_min_penalty": 0.0}, "timeout_min_penalty"),
            ({"stage_scale_floor": 1.5}, "stage_scale_floor"),
            ({"stage_scale_floor": 0.0}, "stage_scale_floor"),
            ({"reverse_motion_penalty": -0.5}, "reverse_motion_penalty"),
            ({"obstacle_speed_power": -1.0}, "obstacle_speed_power"),
            ({"failure_dense_refund_cap": -1.0}, "failure_dense_refund_cap"),
            ({"failure_dense_cap": -0.1}, "failure_dense_cap"),
            ({"failure_time_compensation": -0.1}, "failure_time_compensation"),
            ({"obstacle_near_goal_factor": -0.1}, "obstacle_near_goal_factor"),
            ({"obstacle_near_goal_range": 0.0}, "obstacle_near_goal_range"),
        ],
    )
    def test_invalid_values_rejected(self, tmp_path, params, needle):
        _write_minimal(tmp_path, reward_params=params)
        with pytest.raises(ValueError, match=needle):
            from_config_dir(tmp_path)

    def test_train_yaml_unknown_env_key_rejected(self, tmp_path):
        _write_minimal(tmp_path)
        (tmp_path / "train.yaml").write_text(
            "env:\n  v_max: 1.2\n  not_an_env_key: 3\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="未知键"):
            from_config_dir(tmp_path)

    def test_train_yaml_wrong_type_rejected(self, tmp_path):
        _write_minimal(tmp_path)
        (tmp_path / "train.yaml").write_text("env:\n  v_max: fast\n", encoding="utf-8")
        with pytest.raises(ValueError, match="类型必须是 float"):
            from_config_dir(tmp_path)


NL = "\n"  # 让下面的 YAML 片段拼接读起来是一行一段


class TestDynamicsSection:
    """robot.dynamics 段（dynamics 三级开关的配置入口）。

    为什么挂在 robot 段而不是新增顶层键：TestPackagedYaml 锁住了
    "env.yaml 顶层键 = {world, robot}"（与主仓一致），而 env.yaml 在 chassis-sim 内
    只被 _build_env_config 按段消费。
    """

    def _write_env(self, tmp_path, dynamics: str, *, reward: bool = True):
        if reward:
            (tmp_path / "reward.yaml").write_text(
                yaml.safe_dump(
                    {"reward_weights": {"cte_penalty": 6.0}, "reward_params": {}}
                ),
                encoding="utf-8",
            )
        (tmp_path / "env.yaml").write_text(
            "world:"
            + NL
            + "  step_time: 0.1"
            + NL
            + "robot:"
            + NL
            + "  shape:"
            + NL
            + "    wheelbase: 0.145"
            + NL
            + dynamics,
            encoding="utf-8",
        )

    def test_packaged_yaml_matches_dataclass_defaults(self):
        """包内 env.yaml 的三级开关默认值必须与 EnvConfig 默认逐位相同。

        from_config_dir(PKG_DIR) == EnvConfig() 已经守着这条（见 TestPackagedYaml），
        这里额外把每个键单独点名，好在改动时直接指出是哪个键漂了。
        """
        raw = yaml.safe_load(
            importlib.resources.files("chassis_sim.config")
            .joinpath("env.yaml")
            .read_text(encoding="utf-8")
        )
        dyn = raw["robot"]["dynamics"]
        assert dyn["level"] == EnvConfig().dynamics == "none"
        assert dyn["accel_max"] == EnvConfig().accel_max
        assert dyn["steer_rate_max"] == EnvConfig().steer_rate_max
        assert dyn["v_tau"] == EnvConfig().v_tau
        assert dyn["omega_tau"] == EnvConfig().omega_tau
        assert dyn["v_deadband"] == EnvConfig().v_deadband
        assert dyn["wheel_slip"] == EnvConfig().wheel_slip

    def test_user_directory_can_select_wheel_level(self, tmp_path):
        self._write_env(
            tmp_path,
            "  dynamics:"
            + NL
            + "    level: wheel"
            + NL
            + "    accel_max: 0.9"
            + NL
            + "    steer_rate_max: 2.5"
            + NL
            + "    v_tau: 0.3"
            + NL
            + "    omega_tau: 0.2"
            + NL
            + "    v_deadband: 0.05"
            + NL
            + "    wheel_slip: 0.1"
            + NL,
        )
        env, _ = from_config_dir(tmp_path)
        assert env.dynamics == "wheel"
        assert env.accel_max == pytest.approx(0.9)
        assert env.steer_rate_max == pytest.approx(2.5)
        assert env.v_tau == pytest.approx(0.3)
        assert env.omega_tau == pytest.approx(0.2)
        assert env.v_deadband == pytest.approx(0.05)
        assert env.wheel_slip == pytest.approx(0.1)

    def test_train_yaml_env_section_wins_over_env_yaml(self, tmp_path):
        """train.yaml 的 env 段仍是主口（历史行为），robot.dynamics 只是缺省来源。"""
        self._write_env(
            tmp_path,
            "  dynamics:" + NL + "    level: limits" + NL + "    accel_max: 0.9" + NL,
        )
        (tmp_path / "train.yaml").write_text(
            "env:" + NL + "  accel_max: 0.4" + NL, encoding="utf-8"
        )
        env, _ = from_config_dir(tmp_path)
        assert env.dynamics == "limits"
        assert env.accel_max == pytest.approx(0.4)

    def test_invalid_level_rejected(self, tmp_path):
        self._write_env(tmp_path, "  dynamics:" + NL + "    level: bogus" + NL)
        # 加载期校验自己给文案（不 import dynamics 模块），键路径写在消息里
        with pytest.raises(ValueError, match="robot.dynamics.level 必须是"):
            from_config_dir(tmp_path)

    @pytest.mark.parametrize(
        "key, value, needle",
        [
            ("v_tau", 0.0, "v_tau"),
            ("omega_tau", -0.2, "omega_tau"),
            ("v_deadband", -0.1, "v_deadband"),
            ("wheel_slip", 1.0, "wheel_slip"),
            ("wheel_slip", -0.05, "wheel_slip"),
        ],
    )
    def test_invalid_numbers_rejected(self, tmp_path, key, value, needle):
        self._write_env(
            tmp_path,
            "  dynamics:" + NL + "    level: wheel" + NL + f"    {key}: {value}" + NL,
        )
        with pytest.raises(ValueError, match=needle):
            from_config_dir(tmp_path)

    def test_section_is_optional(self, tmp_path):
        """没有 robot.dynamics 段 ⇒ 落回 dataclass 默认（旧目录继续能读）。"""
        (tmp_path / "reward.yaml").write_text(
            yaml.safe_dump(
                {"reward_weights": {"cte_penalty": 6.0}, "reward_params": {}}
            ),
            encoding="utf-8",
        )
        (tmp_path / "env.yaml").write_text(
            "world:"
            + NL
            + "  step_time: 0.1"
            + NL
            + "robot:"
            + NL
            + "  shape:"
            + NL
            + "    wheelbase: 0.145"
            + NL,
            encoding="utf-8",
        )
        env, _ = from_config_dir(tmp_path)
        assert env.dynamics == "none"
        assert env.v_tau == EnvConfig().v_tau
