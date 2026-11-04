"""EnvConfig YAML 快照（to_yaml / from_yaml）的行为锁定。

快照是"一次运行可完整复现"的载体，不是 DSL：全部字段逐字段写出，
sensors / obs_spec 用带类型标签的 dict 描述，from_yaml 重建等价对象后
dataclass 相等必须成立。
"""

import numpy as np
import pytest
import yaml

from chassis_sim import (
    Act,
    EnvConfig,
    Goal,
    Imu,
    Lidar,
    LidarSegment,
    ObsSpec,
    Odom,
    PathSegment,
    WheelOdom,
)


def _rich_cfg():
    """覆盖三个需要特殊序列化的字段 + 若干标量。"""
    return EnvConfig(
        chassis="mecanum",
        v_max=0.8,
        dt=0.05,
        max_steps=77,
        path_fill_mode="hold",
        sensor_seed=11,
        emit_reward_components=False,
        world_bounds=(0.0, 5.0, 0.0, 5.0),
        scenario="corridor",
        sensors=(
            Lidar(name="front", mount=(0.2, 0.0, 0.1), n_rays=90, max_range=6.0),
            Imu("imu", noise_std=0.01),
            WheelOdom("wheel", track_width=0.3),
        ),
        obs_spec=ObsSpec(
            (
                Odom(),
                PathSegment(points=10),
                Goal(),
                Act(action_dim=3),
                LidarSegment(
                    name="front",
                    sensors=("front",),
                    pool="max",
                    ray_counts={"front": 90},
                ),
            )
        ),
    )


class TestRoundTrip:
    def test_rich_config_is_dataclass_equal(self, tmp_path):
        cfg = _rich_cfg()
        path = cfg.to_yaml(tmp_path / "snap.yaml")
        back = EnvConfig.from_yaml(path)
        assert back == cfg
        assert back.chassis == "mecanum"
        assert back.max_steps == 77
        assert back.world_bounds == (0.0, 5.0, 0.0, 5.0)

    def test_defaults_roundtrip(self, tmp_path):
        cfg = EnvConfig()
        back = EnvConfig.from_yaml(cfg.to_yaml(tmp_path / "default.yaml"))
        assert back == cfg
        assert back.sensors is None
        assert back.obs_spec is None

    def test_repeated_reads_are_stable(self, tmp_path):
        cfg = _rich_cfg()
        path = cfg.to_yaml(tmp_path / "snap.yaml")
        assert EnvConfig.from_yaml(path) == EnvConfig.from_yaml(path) == cfg

    def test_yaml_is_human_readable(self, tmp_path):
        path = _rich_cfg().to_yaml(tmp_path / "snap.yaml")
        text = path.read_text(encoding="utf-8")
        assert "chassis: mecanum" in text
        for tag in ("type: Lidar", "type: Imu", "type: WheelOdom", "segments:"):
            assert tag in text


class TestSensorsRebuild:
    def test_types_and_fields(self, tmp_path):
        back = EnvConfig.from_yaml(_rich_cfg().to_yaml(tmp_path / "s.yaml"))
        lidar, imu, odom = back.sensors
        assert isinstance(lidar, Lidar)
        assert isinstance(imu, Imu)
        assert isinstance(odom, WheelOdom)
        assert lidar.name == "front"
        assert lidar.mount == (0.2, 0.0, 0.1)
        assert lidar.n_rays == 90 and lidar.max_range == 6.0
        assert imu.name == "imu" and imu.noise_std == 0.01
        assert odom.track_width == 0.3

    def test_empty_tuple_roundtrip(self, tmp_path):
        cfg = EnvConfig(sensors=())
        back = EnvConfig.from_yaml(cfg.to_yaml(tmp_path / "empty.yaml"))
        assert back.sensors == () and back == cfg

    def test_unknown_sensor_type_is_loud(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.safe_dump({"sensors": [{"type": "Radar"}]}, allow_unicode=True),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="未知的传感器描述"):
            EnvConfig.from_yaml(path)

    def test_unknown_sensor_field_is_loud(self, tmp_path):
        path = tmp_path / "bad2.yaml"
        payload = {"sensors": [{"type": "Imu", "name": "imu", "gyro_std": 0.1}]}
        path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="未知字段"):
            EnvConfig.from_yaml(path)


class TestObsSpecRebuild:
    def test_segments_and_dims(self, tmp_path):
        back = EnvConfig.from_yaml(_rich_cfg().to_yaml(tmp_path / "o.yaml"))
        spec = back.obs_spec
        assert isinstance(spec, ObsSpec)
        assert [type(s).__name__ for s in spec.segments] == [
            "Odom",
            "PathSegment",
            "Goal",
            "Act",
            "LidarSegment",
        ]
        lidar_seg = spec.segments[-1]
        assert lidar_seg.ray_counts == {"front": 90}
        assert lidar_seg.pool == "max"
        assert spec.resolved().describe() == back.obs_spec.resolved().describe()
        assert spec.resolved().total_dim == 4 + 40 + 2 + 3 + 90

    def test_default_spec_roundtrip(self, tmp_path):
        cfg = EnvConfig(obs_spec=ObsSpec.default(action_dim=3, path_points=12))
        back = EnvConfig.from_yaml(cfg.to_yaml(tmp_path / "d.yaml"))
        assert back == cfg
        assert back.obs_spec.resolved().total_dim == 4 + 48 + 2 + 3

    def test_custom_segment_refuses_to_serialize(self, tmp_path):
        from chassis_sim import Custom

        cfg = EnvConfig(
            obs_spec=ObsSpec((Odom(), Custom("extra", 2, lambda ctx: np.zeros(2))))
        )
        with pytest.raises(ValueError, match="Custom"):
            cfg.to_yaml(tmp_path / "custom.yaml")

    def test_unknown_segment_tag_is_loud(self, tmp_path):
        path = tmp_path / "seg.yaml"
        payload = {"obs_spec": {"segments": [{"type": "Thermometer"}]}}
        path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="未知的观测段类型标签"):
            EnvConfig.from_yaml(path)


class TestUnknownKeys:
    def test_unknown_top_level_key_names_itself(self, tmp_path):
        path = tmp_path / "unknown.yaml"
        path.write_text(
            yaml.safe_dump({"v_max": 1.0, "turbo": True}, allow_unicode=True),
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as excinfo:
            EnvConfig.from_yaml(path)
        message = str(excinfo.value)
        assert "turbo" in message
        n_fields = len(EnvConfig.__dataclass_fields__)
        assert str(n_fields) in message
