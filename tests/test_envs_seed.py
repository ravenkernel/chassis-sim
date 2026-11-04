"""种子语义（1.1.0 D）：seed 控制域随机化、sensor_seed 派生规则、注入桥不接管随机源。

三条契约：

1. `EnvConfig.seed` 是**运行**种子；`sensor_seed` 未指定时跟随它（规则唯一实现
   `config.schema.resolve_sensor_seed`），两者都没有 ⇒ 0（1.0.x 的默认值）。
2. `SimpleBridge(seed=...)` 自持一条 Generator ⇒ `domain_rand=True` 的 LiDAR 噪声
   与里程计漂移同 seed 逐位可复现；`seed=None` 仍吃全局 `np.random`（历史行为）。
3. env 只把 seed 传给**自己造的**桥；注入 `bridge=` 时 env 不读也不写它的随机源。
"""

from __future__ import annotations

import numpy as np
import pytest

import chassis_sim as cs
from chassis_sim.bridges import SimpleBridge
from chassis_sim.chassis import AckermannChassis
from chassis_sim.config.schema import EnvConfig, resolve_sensor_seed
from chassis_sim.envs import ChassisSimEnv

_PATH = np.column_stack([np.linspace(1.0, 6.0, 40), np.full(40, 2.0), np.zeros(40)])
_OBSTACLES = [
    {"shape": {"name": "circle", "radius": 0.30}, "state": [3.0, 2.5, 0.0]},
    {
        "shape": {"name": "rectangle", "length": 0.60, "width": 0.40},
        "state": [4.2, 1.4, 0.4],
    },
]


def _bridge(*, seed: int | None, domain_rand: bool = True) -> SimpleBridge:
    """一台阿克曼桥；seed=None ⇒ 域随机化吃全局 np.random。"""
    return SimpleBridge(AckermannChassis(), domain_rand=domain_rand, seed=seed)


def _trace(
    bridge: SimpleBridge, steps: int = 6, *, ray_noise: bool = True
) -> list[np.ndarray]:
    """reset 后逐步取 [odom, 单次雷达] —— 域随机化的两条随机流都覆盖。"""
    bridge.reset(
        obstacles=[dict(o) for o in _OBSTACLES],
        path_waypoints=_PATH.copy(),
        initial_heading=0.0,
    )
    out: list[np.ndarray] = []
    for _ in range(steps):
        bridge.step(np.array([0.05, 0.5]))
        out.append(bridge.get_odometry().copy())
        # 雷达域随机化只在 _cast_rays(domain_rand=True) 上发生（参数化传感器走
        # 自己的 Generator），因此这里直接调它——1.0.x 的唯一入口。
        out.append(
            bridge._cast_rays(
                front=True,
                n_rays=8,
                fov=float(np.pi),
                max_range=10.0,
                range_min=0.05,
                domain_rand=ray_noise,
            )
        )
    return out


def _same(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


class TestSeedDerivation:
    """resolve_sensor_seed 是传感器种子推导的唯一实现点。"""

    @pytest.mark.parametrize(
        ("sensor_seed", "seed", "expected"),
        [
            (None, None, 0),  # 都没给 ⇒ 1.0.x 的默认值 0
            (3, None, 3),  # 只给 sensor_seed ⇒ 原样
            (None, 3, 3),  # 只给 seed ⇒ 等价于 1.0.x 的 sensor_seed=3
            (3, 4, 3),  # 两者都给 ⇒ sensor_seed 优先（各管各的，不冲突）
        ],
    )
    def test_rule(self, sensor_seed, seed, expected):
        assert resolve_sensor_seed(sensor_seed, seed) == expected

    def test_config_defaults_are_none(self):
        cfg = EnvConfig()
        assert cfg.seed is None
        assert cfg.sensor_seed is None

    def test_config_seed_roundtrips_through_yaml(self, tmp_path):
        cfg = EnvConfig(seed=4)
        path = cfg.to_yaml(tmp_path / "snap.yaml")
        assert EnvConfig.from_yaml(path) == cfg


class TestBridgeSeed:
    """seed 决定域随机化噪声的随机源；None 时行为与 1.0.x 逐位相同。"""

    def test_same_seed_is_bitwise_reproducible(self):
        a = _trace(_bridge(seed=42))
        b = _trace(_bridge(seed=42))
        assert _same(a, b)

    def test_other_seed_differs(self):
        a = _trace(_bridge(seed=42))
        c = _trace(_bridge(seed=43))
        assert not _same(a, c)

    def test_domain_rand_off_is_unaffected_by_seed(self):
        # 关掉域随机化时 seed 一个数都不许动（里程计漂移与雷达噪声都不执行）
        a = _trace(_bridge(seed=42, domain_rand=False), ray_noise=False)
        b = _trace(_bridge(seed=None, domain_rand=False), ray_noise=False)
        assert _same(a, b)

    def test_seeded_bridge_does_not_consume_global_random(self):
        np.random.seed(7)
        expected = np.random.random(3)
        np.random.seed(7)
        _trace(_bridge(seed=42))
        assert np.array_equal(np.random.random(3), expected)

    def test_seed_none_still_follows_global_random(self):
        np.random.seed(123)
        a = _trace(_bridge(seed=None))
        np.random.seed(123)
        b = _trace(_bridge(seed=None))
        assert _same(a, b)
        np.random.seed(124)
        c = _trace(_bridge(seed=None))
        assert not _same(a, c)


class TestEnvSeed:
    """env 层：seed 三层解析 + 只给自己造的桥 + 不给 seed 时逐位不变。"""

    def test_make_seed_reaches_own_bridge(self):
        env = cs.make("ackermann", domain_rand=True, seed=42)
        assert env.seed == 42
        assert env.bridge._seed == 42

    def test_cfg_field_is_the_second_layer(self):
        env = cs.make(EnvConfig(seed=9))
        assert env.seed == 9
        assert env.sensor_seed == 9
        # kwargs 覆盖 cfg
        assert cs.make(EnvConfig(seed=9), seed=11).seed == 11

    def test_sensor_seed_follows_seed_only_when_absent(self):
        assert cs.make("ackermann", seed=3).sensor_seed == 3
        assert cs.make("ackermann", sensor_seed=5).sensor_seed == 5
        both = cs.make("ackermann", seed=3, sensor_seed=5)
        assert (both.seed, both.sensor_seed) == (3, 5)

    def test_domain_rand_is_reproducible(self):
        acts = [np.array([0.4, 0.6], dtype=np.float32)] * 20

        def run(seed):
            env = cs.make("ackermann", domain_rand=True, seed=seed, max_steps=30)
            obs, _ = env.reset(seed=0)
            out = [obs.copy()]
            for a in acts:
                obs, *_ = env.step(a)
                out.append(obs.copy())
            env.close()
            return out

        same = run(42)
        again = run(42)
        other = run(43)
        assert all(np.array_equal(x, y) for x, y in zip(same, again))
        assert not all(np.array_equal(x, y) for x, y in zip(same, other))

    def test_seed_does_not_change_domain_rand_off(self):
        acts = [np.array([0.4, 0.6], dtype=np.float32)] * 10

        def run(seed):
            env = cs.make("ackermann", domain_rand=False, seed=seed, max_steps=30)
            obs, _ = env.reset(seed=0)
            out = [obs.copy()]
            for a in acts:
                obs, *_ = env.step(a)
                out.append(obs.copy())
            env.close()
            return out

        plain = run(None)
        seeded = run(42)
        assert all(np.array_equal(x, y) for x, y in zip(plain, seeded))

    def test_injected_bridge_keeps_its_own_random_source(self):
        chassis = AckermannChassis()
        bridge = SimpleBridge(chassis, domain_rand=True)  # 不给 seed
        env = ChassisSimEnv(chassis=chassis, bridge=bridge, seed=42)
        # env 记住了解析出的运行种子，但**不**接管注入桥的随机源
        assert env.seed == 42
        assert bridge._seed is None
        assert bridge._rng is None
