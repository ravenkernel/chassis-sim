"""action_space 规格 + 入口校验（clip / 形状错误抛 ValueError）。"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from chassis_sim.chassis import (
    AckermannChassis,
    DiffDriveChassis,
    MecanumChassis,
    available_chassis,
    make_chassis,
)

# 规格（三个车型逐位比对）
SPEC = {
    "ackermann": {
        "shape": (2,),
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
    },
    "diff_drive": {
        "shape": (2,),
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
    },
    "mecanum": {
        "shape": (3,),
        "low": [-1.0, -1.0, -1.0],
        "high": [1.0, 1.0, 1.0],
    },
}


@pytest.mark.parametrize("name", sorted(SPEC))
def test_action_space_matches_spec(name):
    ch = make_chassis(name)
    space = ch.action_space()

    assert isinstance(space, spaces.Box)
    assert tuple(space.shape) == SPEC[name]["shape"]
    np.testing.assert_array_equal(
        np.asarray(space.low, dtype=np.float64), np.asarray(SPEC[name]["low"])
    )
    np.testing.assert_array_equal(
        np.asarray(space.high, dtype=np.float64), np.asarray(SPEC[name]["high"])
    )
    assert ch.action_dim == SPEC[name]["shape"][0]
    assert ch.name == name


def test_registry_contents():
    assert available_chassis() == ["ackermann", "diff_drive", "mecanum"]
    assert isinstance(make_chassis("mecanum"), MecanumChassis)
    assert isinstance(make_chassis("ackermann"), AckermannChassis)
    assert isinstance(make_chassis("diff_drive"), DiffDriveChassis)


def test_registry_unknown_name_raises_keyerror_listing_available():
    with pytest.raises(KeyError) as excinfo:
        make_chassis("tank")
    msg = str(excinfo.value)
    assert "tank" in msg
    for name in ("ackermann", "diff_drive", "mecanum"):
        assert name in msg


# ── 越界 action 被 clip ──────────────────────────────────────────────


def test_ackermann_out_of_range_action_is_clipped_then_mapped():
    ch = AckermannChassis(
        delta_max=0.526, v_min=-1.0, v_max=1.0, curvature_action=False
    )
    cmd = ch.command(np.array([5.0, 3.0]))
    assert cmd.v == 1.0  # v 夹到 v_max
    assert cmd.omega == pytest.approx(1.0 / 0.145 * np.tan(0.526), rel=1e-15)

    cmd = ch.command(np.array([-5.0, -3.0]))
    assert cmd.v == -1.0  # v 夹到 v_min
    assert cmd.omega == pytest.approx(-1.0 / 0.145 * np.tan(-0.526), rel=1e-15)


def test_diff_drive_out_of_range_action_is_clipped():
    ch = DiffDriveChassis(v_max=1.0, omega_max=1.0)
    cmd = ch.command(np.array([5.0, -9.0]))
    assert (cmd.v, cmd.omega, cmd.vy) == (1.0, -1.0, 0.0)


def test_mecanum_out_of_range_action_is_clipped():
    ch = MecanumChassis(v_max=1.0, vy_max=0.5, omega_max=1.0)
    cmd = ch.command(np.array([9.0, 9.0, 9.0]))
    assert (cmd.v, cmd.vy, cmd.omega) == (1.0, 0.5, 1.0)
    cmd = ch.command(np.array([-9.0, -9.0, -9.0]))
    assert (cmd.v, cmd.vy, cmd.omega) == (-1.0, -0.5, -1.0)


# ── 形状不符抛 ValueError（消息含期望/实际维度）────────────────────


@pytest.mark.parametrize(
    "name,bad_action",
    [
        ("ackermann", np.zeros(1)),
        ("ackermann", np.zeros(3)),
        ("diff_drive", np.zeros(3)),
        ("mecanum", np.zeros(2)),
        ("mecanum", np.zeros((1, 3))),
    ],
)
def test_wrong_shape_raises_valueerror_with_both_dims(name, bad_action):
    ch = make_chassis(name)
    expected = ch.action_space().shape
    with pytest.raises(ValueError) as excinfo:
        ch.command(bad_action)
    msg = str(excinfo.value)
    assert str(bad_action.shape) in msg, msg
    assert str(tuple(expected)) in msg, msg


# ── describe() 契约（控制器层依赖，键名不许改名）──────────────────────

DESCRIBE_KEYS = {
    "name",
    "action_dim",
    "action_space",
    "footprint",
    "lidar_mounts",
    "limits",
    "params",
}
LIMIT_KEYS = {"v_min", "v_max", "omega_max", "vy_max"}
PARAM_CONTRACT_KEYS = {"wheelbase", "delta_max", "curvature_action", "kappa_max"}


@pytest.mark.parametrize("name", sorted(SPEC))
def test_describe_contract_shape_and_structure(name):
    ch = make_chassis(name)
    d = ch.describe()

    assert DESCRIBE_KEYS <= set(d)
    assert d["name"] == name
    assert d["action_dim"] == SPEC[name]["shape"][0]
    np.testing.assert_array_equal(
        np.asarray(d["action_space"]["low"], dtype=np.float64),
        np.asarray(SPEC[name]["low"]),
    )
    np.testing.assert_array_equal(
        np.asarray(d["action_space"]["high"], dtype=np.float64),
        np.asarray(SPEC[name]["high"]),
    )
    assert d["footprint"] == [0.38, 0.2]
    assert d["lidar_mounts"] == [[0.145, 0.0, 0.0], [-0.155, 0.0, float(np.pi)]]
    # limits：四个键齐全、全是 float（不适用的车型给 0.0）
    assert set(d["limits"]) == LIMIT_KEYS
    assert all(isinstance(v, float) for v in d["limits"].values())
    # params：契约键齐全（不适用者 None）
    assert PARAM_CONTRACT_KEYS <= set(d["params"])


def test_describe_ackermann_limits_and_params_match_constructor():
    ch = AckermannChassis(
        wheelbase=0.145,
        delta_max=0.526,
        v_min=-1.0,
        v_max=1.0,
        curvature_action=True,
        kappa_max=4.0,
    )
    lim, par = ch.describe()["limits"], ch.describe()["params"]
    assert (lim["v_min"], lim["v_max"], lim["vy_max"]) == (-1.0, 1.0, 0.0)
    assert lim["omega_max"] == pytest.approx(1.0 / 0.145 * np.tan(0.526), rel=1e-15)
    assert par["wheelbase"] == 0.145 and par["delta_max"] == 0.526
    assert par["curvature_action"] is True and par["kappa_max"] == 4.0
    assert (par["v_min"], par["v_max"]) == (-1.0, 1.0)


def test_describe_diff_drive_limits_and_params_match_constructor():
    ch = DiffDriveChassis(wheelbase=0.145, v_max=0.8, omega_max=1.2)
    lim, par = ch.describe()["limits"], ch.describe()["params"]
    assert (lim["v_min"], lim["v_max"], lim["omega_max"], lim["vy_max"]) == (
        -0.8,
        0.8,
        1.2,
        0.0,
    )
    assert par["wheelbase"] == 0.145
    assert (par["delta_max"], par["curvature_action"], par["kappa_max"]) == (
        None,
        None,
        None,
    )


def test_describe_mecanum_limits_and_params_match_constructor():
    ch = MecanumChassis(lx=0.2, ly=0.12, v_max=1.0, vy_max=0.5, omega_max=0.8)
    lim, par = ch.describe()["limits"], ch.describe()["params"]
    assert (lim["v_min"], lim["v_max"], lim["omega_max"], lim["vy_max"]) == (
        -1.0,
        1.0,
        0.8,
        0.5,
    )
    assert (par["lx"], par["ly"], par["v_max"], par["vy_max"], par["omega_max"]) == (
        0.2,
        0.12,
        1.0,
        0.5,
        0.8,
    )
    assert (
        par["wheelbase"],
        par["delta_max"],
        par["curvature_action"],
        par["kappa_max"],
    ) == (
        None,
        None,
        None,
        None,
    )


def test_mecanum_action_space_is_normalized_whatever_the_physical_limits():
    ch = MecanumChassis(v_max=0.4, vy_max=0.3, omega_max=0.7)
    space = ch.action_space()
    np.testing.assert_array_equal(
        space.low, np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        space.high, np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    cmd = ch.command(np.array([1.0, 1.0, 1.0]))
    assert (cmd.v, cmd.vy, cmd.omega) == (0.4, 0.3, 0.7)
    cmd = ch.command(np.array([-1.0, -1.0, -1.0]))
    assert (cmd.v, cmd.vy, cmd.omega) == (-0.4, -0.3, -0.7)


def test_diff_drive_action_space_is_normalized_whatever_the_physical_limits():
    """差速与阿克曼/麦轮同一条契约：动作恒为 [-1,1]^2，物理量只在 command() 内产生。"""
    ch = DiffDriveChassis(v_max=1.2, omega_max=0.8)
    space = ch.action_space()
    np.testing.assert_array_equal(space.low, np.array([-1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(space.high, np.array([1.0, 1.0], dtype=np.float32))
    cmd = ch.command(np.array([1.0, 1.0]))
    assert (cmd.v, cmd.omega, cmd.vy) == (1.2, 0.8, 0.0)
    cmd = ch.command(np.array([-1.0, -1.0]))
    assert (cmd.v, cmd.omega, cmd.vy) == (-1.2, -0.8, 0.0)
    cmd = ch.command(np.array([0.5, 0.5]))
    assert (cmd.v, cmd.omega) == (0.6, 0.4)
