"""EnvConfig / RewardConfig —— env 层自带的配置 dataclass。

从主仓 training/config.py 抽出 EnvConfig（原 760-807 行）与 RewardConfig
（原 810-938 行）两个 dataclass，并把原先散落在 TrainingConfig.from_config_dir
（原 976-1425 行）里的 env / reward 装配逻辑一并搬过来 —— 抽包后本库不再向
上依赖 training/。

与主仓的三处有意差异：

1. YAML 用 **pyyaml** 而不是 omegaconf（本包的 core 依赖里没有 omegaconf）。
   原实现只用到 OmegaConf 的三件事，逐条等价替换：
     · OmegaConf.load + 点路径访问      → yaml.safe_load + _get_nested() 查找；
     · OmegaConf.structured(schema) + merge + to_object 的结构校验
                                        → _validate_structured()：未知键拒绝 +
                                          类型粗校验 + 缺省键落回 dataclass
                                          默认值（与 structured 的补全语义一致）；
     · _flatten_omega 扁平化            → 本包只消费固定几个键，不再需要。
   **没有**用到插值（美元符号加大括号的写法）、配置组合（defaults: 段）、
   命令行覆盖这几项，因此不存在需要等价实现的语义；OmegaConf 的 EnumNode
   按成员名查找这一行为也不涉及（本包不读枚举键）。
2. 包里不发布 train.yaml（它是训练配置）。train.yaml 缺失时，原本从
   train.yaml 读的 EnvConfig 字段一律落回 dataclass 字段默认值；train.yaml
   存在时（例如用户指着主仓的 config/ 目录）仍逐位沿用主仓读法。
3. 默认配置通过 importlib.resources 定位，**不依赖 CWD** —— 这是主仓
   training/config.py:976-1001 里 Path(config_dir) + open() 那个 bug 的修复。

数值的唯一权威仍是 YAML；YAML 缺键时的兜底值一律取 dataclass 字段默认值
（不写字面量），以免出现"第三套默认值"（主仓 P2-13 的教训）。
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    """Environment physics + privileged + domain_rand + display (~25 fields)."""

    delta_max: float = 0.526
    kappa_max: float = 4.0
    curvature_action: bool = False
    v_min: float = -0.6
    v_max: float = 1.2
    goal_dist_threshold: float = 0.3
    goal_heading_tolerance: float = 0.6
    goal_end_wp_count: int = 6
    goal_end_wp_margin: int = 2
    path_deviation_limit: float = 4.0
    accel_max: float = 0.0
    steer_rate_max: float = 0.0
    dt: float = 0.1
    wheelbase: float = 0.145
    lidar_max_range: float = 10.0
    privileged_enabled: bool = False
    privileged_bins: int = 36
    privileged_range: float = 5.0
    domain_rand_enabled: bool = True
    # ── 域随机化强度（主仓 2026-10-05 接线）──────────────────────────────
    # 这三个值在 NumpySimBridge / SimBridge 的构造器里一直存在，但**所有**
    # create_bridge 调用点都只传 domain_rand(bool)，从没传过它们；
    # train.yaml 的 domain_rand.lidar_noise_std 等键此前也没有任何代码读取
    # ⇒ "域随机化强度的唯一旋钮"是死的：Python 默认值恰好等于 YAML 值
    # (0.03/0.01/0.005)，改 YAML 完全不生效。现纳入 EnvConfig 并端到端下发。
    lidar_noise_std: float = 0.03  # LiDAR 高斯噪声标准差 (m)
    odom_drift_xy: float = 0.01  # 里程计 XY 漂移系数（每步随机游走）
    odom_drift_theta: float = 0.005  # 里程计角度漂移系数（每步随机游走）
    envs_panel_enabled: bool = True
    curriculum_enabled: bool = True
    # ── 档位投票滞后 (2026-10-06) ────────────────────────────────────────
    # env 内部"该正着开还是倒着开"的稳定档位需要的连续同向票数。
    # raw path_dir 在 cusp/自接近区每步抖动 27%~32%（真 stage3/4 实测）
    # ⇒ 直接当奖励闸门 = 惩罚噪声。1 = 关闭（逐位回到旧行为）。
    # 只作用于奖励侧（闸门 + 倒车段偏航权重）与 info，**不改观测**（494 维）。
    gear_lag_votes: int = 3
    # ── 每步诊断字段的生产开关（主仓 P2-22 接线，2026-10-23）─────────────
    # AckermannEnv 早就有 emit_reward_components 参数（默认 True），但**没有
    # 任何调用点传过它** ⇒ 生产永远拿默认值，"省内存"的旋钮一直是死键。
    # reward_components 实测 1711 B/条，训练路径零消费者；置 False 可省 ≈1.1 GB。
    # 默认 True = 行为不变（诊断/TUI 仍可看到分量）。
    emit_reward_components: bool = True
    # ── 动力学三级开关（2026-11-04）──────────────────────────────────────
    # "none"   纯运动学。与加入本开关前**逐位相同**（dynamics.py 连 import 都不发生）。
    # "limits" accel_max / steer_rate_max 硬限幅；两者为 0 时用 envs/dynamics.py 的
    #          默认值（1.5 m/s² / 3.0 rad/s），显式给了非零值以用户值为准。
    # "wheel"  limits + 执行器一阶滞后（v_tau / omega_tau）+ 死区 + 轮滑。
    # 取值的权威来源是 env.yaml 的 robot.dynamics 段；train.yaml 的 env.accel_max /
    # env.steer_rate_max 仍是前两个值的**主口**（本包不发布 train.yaml），0/缺失时
    # 才回落到 robot.dynamics 段。语义见 envs/dynamics.py 的模块 docstring。
    dynamics: str = "none"
    v_tau: float = 0.15  # s，速度一阶滞后时间常数（仅 wheel 档）
    omega_tau: float = 0.10  # s，角速度一阶滞后时间常数（仅 wheel 档）
    v_deadband: float = 0.0  # m/s，|v_cmd| 低于此值输出归零（0 = 关，仅 wheel 档）
    wheel_slip: float = 0.0  # 轮滑系数 ∈[0,1)，地速 = 轮速×(1−slip)（0 = 关，仅 wheel）

    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> EnvConfig:
        """只取 env 侧的一半（见模块级 from_config_dir）。"""
        return from_config_dir(config_dir)[0]


@dataclass
class RewardConfig:
    """All reward weights and numerical parameters (~25 fields).

    0.1.1：删除 4 个主仓历史遗留的**死奖励参数** —— RewardCalculator 的签名与
    构造体里的 kwargs.get 都不消费它们，传进来只会被 **kwargs 吞掉，env 侧因此
    在构造期打一条 [WARN][env]：alive_bonus / reverse_step_penalty /
    curvature_scale / heading_alpha。（同批告警的另 8 个键从来不在本 dataclass：
    4 个是 chassis_env.py 里写死的字面量，3 个曾是 EnvConfig 的字段、已一并删除，
    path_d_max 则**保留**——它另有 ObservationBuilder 这条活通路。）
    告警护栏本身**保留**（将来再加死键仍会 warn），见
    tests/test_envs_config.py::TestNoDeadRewardKwargs。
    """

    # ⚠ 本字典必须与 config/reward.yaml 的 reward_weights **逐键逐值**一致
    #   （主仓 tests/test_config_default_sync.py 锁住；改 YAML 就必须同步改这里）。
    #   这份默认值只在"不经 from_config_dir 的构造"里生效（reward_cfg=None），
    #   而它以前长期是另一套值（drive 7.0 / cte 2.5 / obstacle 1.0、且缺
    #   steer_penalty / goal_proximity / heading_penalty_reverse）——
    #   于是 reward_cfg=None 会静默跑一套"生产里从未出现"的奖励（P2-13）。
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "drive": 0.0,  # 已删除的分量（reward.py 不读），留键仅为向后兼容
            "cte_penalty": 6.0,
            "heading_penalty": 0.3,
            "heading_penalty_reverse": 0.5,
            "obstacle": 1.5,
            "progress": 4.0,
            "steer_penalty": 0.5,
            "collision": 1.0,
            "goal": 1.0,
            "goal_heading": 1.0,
            "alive": 0.0,
            "stuck": 1.0,
            "timeout": 1.0,
            "path_deviation": 1.0,
            "goal_proximity": 2.0,
        }
    )
    dense_scale: float = 1.0
    # CTE 的**观测归一化除数**（err 分量第 0 维 = clip(cte/path_d_max, ±1)）。
    # ⚠ 它不是奖励参数（RewardCalculator 从不消费），但 env 侧确实把它转发给
    # ObservationBuilder（chassis_env.py 的 path_d_max=）⇒ 0.1.1 保留本字段与
    # reward.yaml 的键，只把它从 _rw_kwargs（奖励 kwargs）里摘掉。
    path_d_max: float = 4.0
    cte_range: float = 4.0
    cte_delta: float = 0.5
    cte_dead_zone: float = 0.2
    # ⚠ 以下标量默认值必须与 config/reward.yaml 的 reward_params **逐位一致**
    #   （主仓 tests/test_config_default_sync.py::TestRewardConfigDefaultsMatchYaml）。
    #   P2-13 (2026-10-23)：此前这里有 4 个字段是第二套值，与 YAML 分叉——
    #     goal_time_bonus 40 vs 0.0、step_penalty 0.0 vs -0.02、
    #     obstacle_range 1.5 vs 0.8、goal_bonus 200 vs 400。
    #   它们只在"不经 from_config_dir 的构造"（reward_cfg=None / RewardConfig()）
    #   里生效，于是那条路径会静默跑一套从未标定过的奖励配置。
    goal_time_bonus: float = 0.0
    step_penalty: float = -0.02
    obstacle_range: float = 0.8
    collision_penalty: float = -100.0
    # ── 撞墙附加罚 + 超时下限（2026-10-23「失败经济学」）─────────────────
    # 终止项：collision = collision_penalty × stage_scale − collision_surcharge
    #         timeout    = min(−timeout_min_penalty,
    #                          timeout_penalty × stage_scale × (1 − arc_completion))
    # ⇒ margin = timeout − collision
    #          = 100·s + surcharge + min(−P, −100·s·(1−ac))
    #          ≥ min(surcharge, 100·stage_scale_floor + surcharge − P)
    #   默认 S=60 / P=30 / floor=0.2 ⇒ margin 恒 ≥ 50，与 stage、与进度无关。
    #   "撞墙"因此在所有课程阶段都严格劣于"超时"（旧实现里 stage0 的撞墙反而比
    #   超时便宜 5 倍，策略学会了用撞车结束回合）。
    #   surcharge = 0 只关掉附加罚（A/B 回滚用），< 0 在**加载期**被拒；
    #   timeout_min_penalty 必须 > 0 —— 它不让低阶段"悬停到超时"的净回报翻正。
    # 三向一致：reward.yaml / 本字段 / envs/reward.py 签名默认。
    collision_surcharge: float = 60.0
    goal_bonus: float = 400.0
    goal_heading_scale: float = 10.0
    stuck_penalty: float = 0.0
    timeout_penalty: float = -100.0
    # 超时基项下限（见上方 margin 推导）：timeout 基项恒 ≤ −P。
    timeout_min_penalty: float = 30.0
    path_deviation_penalty: float = -100.0
    goal_proximity_weight: float = 0.2
    # ── 失败时刻补偿（2026-10-05 接线）────────────────────────────────
    # 这两个键 reward.py 早就在读，但 reward.yaml 经本 dataclass **白名单**转发，
    # 未列入的键会被静默丢弃 ⇒ 此前"只改 YAML 无效"。
    #   failure_dense_cap: 失败回合稠密累计量的**每步额度**（上限 C=该值×max_steps；
    #                      |Σdense| 超过 C 的部分在终止时退回）。0 = 关闭。
    #   failure_time_compensation: 时间税补回倍率；实测方向与目标相反，默认 0.0。
    # 默认值必须与 envs/reward.py 的签名默认**以及 reward.yaml** 逐位一致（都有测试锁）。
    # 【2026-10-23 诊断后定值】三方统一到 0.12 / 120.0（不是 0.0）：
    # run100 的 collision −57.2(37 步) vs timeout −302.1(300 步) 证明"早死更便宜"
    # 已实测发生（5.3 倍），故重新开启退款并给出单次退款上限 120
    # （终止奖励上界 = 基项 + 120×stage_scale，不再是 +3562 的尖峰）。
    failure_dense_cap: float = 0.12
    failure_dense_refund_cap: float = 120.0
    failure_time_compensation: float = 0.0
    # ── 终点附近的 obstacle 项（P2-14，2026-10-23 从 reward.py 的 kwargs 提上来）──
    # reward.py 在 near_goal（距终点 < 2m）内对 obstacle 项做门控，但门控参数此前
    # 只能从 RewardCalculator(**kwargs) 传，而 RewardConfig 白名单里没有它、
    # reward.yaml 也没有 ⇒ 恒等于 0，行为固定为"终点附近 obstacle 整项清零"。
    #   obstacle_near_goal_factor: 0.0 = 整项清零（当前行为，默认不变）；
    #                              >0  = 乘 max(factor, goal_dist/range) 的线性衰减下限。
    #   obstacle_near_goal_range:  上述线性衰减的分母（m）。
    obstacle_near_goal_factor: float = 0.0
    obstacle_near_goal_range: float = 2.0
    # ── stage_scale 的下限（2026-10-23）────────────────────────────────
    # 失败惩罚（碰撞/卡死/偏离）随课程阶段放大，起步值是 clip 的下限：
    #   stage_scale = clip(stage/3, stage_scale_floor, 1.0)
    # 默认 0.2 = reward.py 此前的硬编码字面量 ⇒ 生产行为**逐位不变**
    # （stage0 碰撞仍为 -20.0）。调大它只抬高**低阶段**的失败代价（stage≥3 恒 1.0）。
    # 三向一致：reward.yaml / 本字段 / RewardCalculator 签名默认。
    stage_scale_floor: float = 0.2
    # ── 速度感知障碍惩罚（2026-10-23）────────────────────────────────
    # 旧公式只看距离 ⇒ 同一位置"全速冲过"与"减速蹭过"罚分完全相同，critic
    # 拿不到任何"该减速"的价格信号（Q 校准审计：92.9% 的状态 argmax 落在
    # a1=+1.0 最大速度边界）。开启后 obstacle 项乘
    #     factor = 1 + (|v| / v_max) ** obstacle_speed_power   ∈ [1, 2]
    #   0.0 = **关闭（默认）** ⇒ 生产行为逐位不变（compute() 里整段跳过）；
    #   >0  = 静止时 factor=1、满速时 factor=2，|v| 单调递增。
    # 校验 ≥ 0（加载期报错）；上限不强制（factor 已被 1+1=2 封顶）。
    # 幂次的真实作用是"把满速端拉陡"：u=1 处恒为 2，u<1 时 p 越大 factor 越贴近 1。
    obstacle_speed_power: float = 0.0
    # ── 前进段倒退的专项定价（2026-10-29）─────────────────────────────
    # 病（run110 最后 20 万步）：起点 27~37% 的确定性动作 v<0；沿路径倒退每步只让
    #   arc_progress 掉 ~0.006（它是路径**比例**）⇒ 4.0×arc_progress ≈ −0.02~−0.04/步。
    #   ⇒「倒退直退到边界墙」几乎免费（step_990080：21 步退 1.9 m 撞墙，总回报 −144.8
    #   优于走满 300 步超时的 −169.6）⇒ 策略主动选择早死。
    # 公式：r_reverse = −w × min(1, |v|/v_max)，**仅当 path_dir>0 且 v<0**。
    #   2.5 = 开启（默认，与 reward.yaml 同值）⇒ 满速倒退 −2.5/步；0.0 = 关闭。
    #   只罚"路径要求前进却倒退"⇒ 倒车段（path_dir<0）上正确倒车不被罚。
    reverse_motion_penalty: float = 2.5

    # ── 便捷构造：从目录读 YAML（与模块级 from_config_dir 同源）──────────
    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> RewardConfig:
        """只取奖励侧的一半（见模块级 from_config_dir）。"""
        return from_config_dir(config_dir)[1]


def _get_nested(d: dict, *keys: str, default: Any = None) -> Any:
    """安全地从嵌套字典中提取值。

    逐层 .get；遇到非 dict 立刻返回 default；末值为 None 时也返回 default
    （与主仓 training/config.py:1428-1436 逐位一致）。
    """
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


# ══════════════════════════════════════════════════════════════════════
# 结构调整校验 —— 替代 OmegaConf.structured(schema) + merge + to_object
# ══════════════════════════════════════════════════════════════════════
# 原实现（主仓 training/config.py:600-618）用结构化 schema 把三件事拒在加载期：
#   ① 未知键（"把 failure_dense_cap 拼错成 failure_dense_cp"曾静默丢弃）；
#   ② 类型不匹配（标量写成 list、bool 写成字符串等）；
#   ③ 缺省键由 schema 的默认值补全（本包用 dataclass 字段默认值兜底，等价）。
# 这里用"类型表 + 未知键差集"逐条等价实现，不做 OmegaConf 的合并/插值。

_WEIGHT_KEYS: tuple[str, ...] = tuple(RewardConfig().reward_weights)

# reward_params 的键名与类型（类型只做粗分类：bool / int / float）
_REWARD_PARAM_TYPES: dict[str, str] = {
    "dense_scale": "float",
    "path_d_max": "float",
    "cte_range": "float",
    "cte_delta": "float",
    "cte_dead_zone": "float",
    "goal_time_bonus": "float",
    "step_penalty": "float",
    "obstacle_range": "float",
    "collision_penalty": "float",
    "collision_surcharge": "float",
    "goal_bonus": "float",
    "goal_heading_scale": "float",
    "stuck_penalty": "float",
    "timeout_penalty": "float",
    "timeout_min_penalty": "float",
    "path_deviation_penalty": "float",
    "goal_proximity_weight": "float",
    "failure_dense_cap": "float",
    "failure_dense_refund_cap": "float",
    "failure_time_compensation": "float",
    "obstacle_near_goal_factor": "float",
    "obstacle_near_goal_range": "float",
    "stage_scale_floor": "float",
    "obstacle_speed_power": "float",
    "reverse_motion_penalty": "float",
}

# 本包会读的 train.yaml 段（train.yaml 里还有 sac_ln/redq/... 等本包不消费的段，
# 因此**只校验这些段内部**的未知键，不做顶层差集 —— 这是与 OmegaConf 全量
# schema 的有意差异，见回报）。
_TRAIN_SECTION_TYPES: dict[str, dict[str, str]] = {
    "env": {
        "delta_max": "float",
        "kappa_max": "float",
        "curvature_action": "bool",
        "v_min": "float",
        "v_max": "float",
        "goal_dist_threshold": "float",
        "goal_heading_tolerance": "float",
        "goal_end_wp_count": "int",
        "goal_end_wp_margin": "int",
        "path_deviation_limit": "float",
        "accel_max": "float",
        "steer_rate_max": "float",
        "gear_lag_votes": "int",
        "emit_reward_components": "bool",
    },
    "privileged": {"enabled": "bool", "n_bins": "int", "max_range": "float"},
    "domain_rand": {
        "enabled": "bool",
        "lidar_noise_std": "float",
        "odom_drift_xy": "float",
        "odom_drift_theta": "float",
    },
    "display": {"envs_panel": "bool"},
    "auto_curriculum": {"enabled": "bool"},
    "path": {"lookahead_distance": "float"},
    "network": {
        "max_path_points": "int",
        "lidar_out": "int",
        "odom_out": "int",
        "path_out": "int",
        "path_channels": "int",
        "path_power": "float",
        "path_fill_mode": "str",
        "path_sentinel_embedding": "bool",
        "path_lookahead_floor": "float",
        "extrapolate_path": "bool",
        "use_geo_features": "bool",
    },
}


def _check_type(value: Any, expect: str, path: str) -> None:
    """粗类型校验（与 OmegaConf structured 的拒绝面等价到"标量类别"级别）。"""
    if expect == "bool":
        ok = isinstance(value, bool)
    elif expect == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif expect == "float":
        # OmegaConf 允许 int 给 float 字段（1 → 1.0），这里同样允许；
        # 但 bool 是 int 的子类，必须显式排除。
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expect == "str":
        ok = isinstance(value, str)
    else:  # pragma: no cover - 类型表写错才会到这里
        raise AssertionError(f"未知的类型标签 {expect!r}（{path}）")
    if not ok:
        raise ValueError(
            f"{path} 类型必须是 {expect}，收到 {type(value).__name__}: {value!r}"
        )


def _validate_section(
    raw: Any, section: str, types: dict[str, str], filename: str
) -> None:
    """校验一个段：形状是 dict、无未知键、已知键类型正确。"""
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError(  # noqa: TRY004
            f"{filename}: {section} 段必须是映射（dict），"
            f"收到 {type(raw).__name__}: {raw!r}"
        )
    unknown = sorted(set(raw) - set(types))
    if unknown:
        raise ValueError(
            f"{filename}: {section} 段存在未知键 {unknown}；允许的键为 {sorted(types)}"
        )
    for key, value in raw.items():
        _check_type(value, types[key], f"{filename}: {section}.{key}")


def _validate_reward_yaml(reward_raw: Any, filename: str = "reward.yaml") -> None:
    """reward.yaml 的结构校验（替代 _RewardYamlSchema）。"""
    if not isinstance(reward_raw, dict):
        raise ValueError(  # noqa: TRY004
            f"{filename}: 顶层必须是映射（dict），收到 {type(reward_raw).__name__}"
        )
    unknown = sorted(set(reward_raw) - {"reward_weights", "reward_params"})
    if unknown:
        raise ValueError(
            f"{filename}: 顶层存在未知键 {unknown}；"
            f"允许的键为 ['reward_params', 'reward_weights']"
        )
    weights = reward_raw.get("reward_weights", {})
    if not isinstance(weights, dict):
        raise ValueError(  # noqa: TRY004
            f"{filename}: reward_weights 必须是映射（dict），"
            f"收到 {type(weights).__name__}"
        )
    bad = sorted(set(weights) - set(_WEIGHT_KEYS))
    if bad:
        raise ValueError(
            f"{filename}: reward_weights 存在未知键 {bad}；"
            f"允许的键为 {list(_WEIGHT_KEYS)}"
        )
    for key, value in weights.items():
        _check_type(value, "float", f"{filename}: reward_weights.{key}")
    _validate_section(
        reward_raw.get("reward_params"), "reward_params", _REWARD_PARAM_TYPES, filename
    )


def _validate_train_yaml(train_raw: Any, filename: str = "train.yaml") -> None:
    """只校验本包消费的段（见 _TRAIN_SECTION_TYPES 的注释）。"""
    if not isinstance(train_raw, dict):
        raise ValueError(  # noqa: TRY004
            f"{filename}: 顶层必须是映射（dict），收到 {type(train_raw).__name__}"
        )
    for section, types in _TRAIN_SECTION_TYPES.items():
        _validate_section(train_raw.get(section), section, types, filename)


def from_config_dir(config_dir: str | Path) -> tuple[EnvConfig, RewardConfig]:
    """从用户给的目录读 YAML，返回 (EnvConfig, RewardConfig)。

    读法：
      · reward.yaml 与 env.yaml **必需**（缺失直接抛 FileNotFoundError，
        与主仓同目录时的行为一致）；
      · train.yaml **存在才读** —— 包里不发布它；缺失时 env / privileged /
        domain_rand / display / auto_curriculum / path / network 各段一律落回
        EnvConfig 字段默认值，其余字段照常从 env.yaml / reward.yaml 装配。
    """
    base = Path(config_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"配置目录不存在: {base}")

    with open(base / "reward.yaml", encoding="utf-8") as f:
        reward_raw: dict[str, Any] = yaml.safe_load(f) or {}
    with open(base / "env.yaml", encoding="utf-8") as f:
        env_raw: dict[str, Any] = yaml.safe_load(f) or {}

    train_path = base / "train.yaml"
    train_raw: dict[str, Any] = {}
    if train_path.is_file():
        with open(train_path, encoding="utf-8") as f:
            train_raw = yaml.safe_load(f) or {}

    # ── 结构化校验（加载期，早在取值之前）──────────────────────────────
    _validate_reward_yaml(reward_raw, "config/reward.yaml")
    if train_raw:
        _validate_train_yaml(train_raw, "config/train.yaml")
    # env.yaml 不做结构校验：world/robot 段由 ir-sim 与桥接按 .get 消费，
    # 还带 world.bounds 之类的可选键（主仓同样不校验它）。

    return _build_env_config(train_raw, reward_raw, env_raw), _build_reward_config(
        reward_raw
    )


def _f(train_raw: dict, default: Any, *keys: str) -> Any:
    """取第一个存在的点路径键，否则返回 default（主仓 _f 的等价实现）。"""
    for key in keys:
        val = _get_nested(train_raw, *key.split("."), default=_MISSING)
        if val is not _MISSING:
            return val
    return default


class _Missing:
    """哨兵：区分"键不存在"与"键的值为 None"。"""

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return "<missing>"


_MISSING = _Missing()


def _build_env_config(train_raw: dict, reward_raw: dict, env_raw: dict) -> EnvConfig:
    """按主仓 TrainingConfig.from_config_dir 的取值顺序装配 EnvConfig。"""
    # ── Env physics（train.yaml 的 env 段；缺失 ⇒ 字段默认值）──
    delta_max = float(_f(train_raw, 0.526, "env.delta_max"))
    kappa_max = float(_f(train_raw, 4.0, "env.kappa_max"))
    curvature_action = bool(_f(train_raw, False, "env.curvature_action"))
    v_min = float(_f(train_raw, -0.6, "env.v_min"))
    v_max = float(_f(train_raw, 1.2, "env.v_max"))
    goal_dist_threshold = float(_f(train_raw, 0.3, "env.goal_dist_threshold"))
    goal_heading_tolerance = float(_f(train_raw, 0.6, "env.goal_heading_tolerance"))
    goal_end_wp_count = int(_f(train_raw, 6, "env.goal_end_wp_count"))
    goal_end_wp_margin = int(_f(train_raw, 2, "env.goal_end_wp_margin"))
    path_deviation_limit = float(_f(train_raw, 4.0, "env.path_deviation_limit"))
    accel_max = float(_f(train_raw, 0.0, "env.accel_max"))
    steer_rate_max = float(_f(train_raw, 0.0, "env.steer_rate_max"))
    gear_lag_votes = int(_f(train_raw, 3, "env.gear_lag_votes"))
    emit_reward_components = bool(_f(train_raw, True, "env.emit_reward_components"))

    # ── Privileged ──
    privileged_enabled = bool(_f(train_raw, False, "privileged.enabled"))
    privileged_bins = int(_f(train_raw, 36, "privileged.n_bins"))
    privileged_range = float(_f(train_raw, 5.0, "privileged.max_range"))

    # ── Domain rand（噪声强度也必须真实读取，否则"改 YAML 无效"）──
    domain_rand_enabled = bool(_f(train_raw, True, "domain_rand.enabled"))
    lidar_noise_std = float(_f(train_raw, 0.03, "domain_rand.lidar_noise_std"))
    odom_drift_xy = float(_f(train_raw, 0.01, "domain_rand.odom_drift_xy"))
    odom_drift_theta = float(_f(train_raw, 0.005, "domain_rand.odom_drift_theta"))

    # ── Display / Auto curriculum ──
    envs_panel_enabled = bool(_f(train_raw, True, "display.envs_panel"))
    curriculum_enabled = bool(_f(train_raw, True, "auto_curriculum.enabled"))

    # ── 观测阈值 obs_danger/obs_safe/stop_threshold 已于 0.1.1 删除 ──────
    # 它们唯一的消费者是 chassis_env 把它们转发给 RewardCalculator，而后者签名
    # 与 kwargs.get 都不读 ⇒ 纯死键（配了等于没配）。现在再写进 reward.yaml 的
    # reward_params 会被当**未知键**拒绝。
    _EC = EnvConfig()  # 兜底值来源（dynamics 段的 level / 时间常数）

    # ── env.yaml：世界步长 / 轮距 / LiDAR 量程（主仓 1308-1317）──
    dt = float(env_raw.get("world", {}).get("step_time", 0.1))
    wheelbase = float(
        _get_nested(env_raw, "robot", "shape", "wheelbase", default=0.145)
    )
    lidar_max_range = 10.0
    for sensor in env_raw.get("robot", {}).get("sensors", []):
        if isinstance(sensor, dict) and sensor.get("type") == "lidar2d":
            lidar_max_range = float(sensor.get("range_max", 10.0))
            break

    # ── env.yaml：robot.dynamics（动力学三级开关，2026-11-04）────────────
    # 挂在 robot 段下而不是新增顶层键：顶层键必须保持 {world, robot}（与主仓一致，
    # 见 tests/test_envs_config.py 的 TestPackagedYaml）；env.yaml 不做结构校验，
    # 该段只被本函数按 .get 消费。
    _dyn_raw = env_raw.get("robot", {}).get("dynamics", {}) or {}
    dynamics = str(_dyn_raw.get("level", _EC.dynamics)).strip().lower()
    if dynamics not in ("none", "limits", "wheel"):
        raise ValueError(
            f"env.yaml 的 robot.dynamics.level 必须是 none/limits/wheel 之一，"
            f"收到 {_dyn_raw.get('level')!r}"
        )
    v_tau = float(_dyn_raw.get("v_tau", _EC.v_tau))
    omega_tau = float(_dyn_raw.get("omega_tau", _EC.omega_tau))
    v_deadband = float(_dyn_raw.get("v_deadband", _EC.v_deadband))
    wheel_slip = float(_dyn_raw.get("wheel_slip", _EC.wheel_slip))
    if not v_tau > 0.0:
        raise ValueError(f"env.yaml 的 robot.dynamics.v_tau 必须 > 0，收到 {v_tau}")
    if not omega_tau > 0.0:
        raise ValueError(
            f"env.yaml 的 robot.dynamics.omega_tau 必须 > 0，收到 {omega_tau}"
        )
    if v_deadband < 0.0:
        raise ValueError(
            f"env.yaml 的 robot.dynamics.v_deadband 必须 >= 0，收到 {v_deadband}"
        )
    if not 0.0 <= wheel_slip < 1.0:
        raise ValueError(
            f"env.yaml 的 robot.dynamics.wheel_slip 必须在 [0, 1) 内，收到 {wheel_slip}"
        )
    # accel_max / steer_rate_max：train.yaml 的 env 段是主口（本包不发布它 ⇒ 通常为
    # 0/缺失），此时再看 env.yaml 的 robot.dynamics 段，仍为 0 就交给
    # dynamics.resolve_limits 按等级取默认值。
    if accel_max == 0.0:
        accel_max = float(_dyn_raw.get("accel_max", 0.0))
    if steer_rate_max == 0.0:
        steer_rate_max = float(_dyn_raw.get("steer_rate_max", 0.0))

    return EnvConfig(
        delta_max=delta_max,
        kappa_max=kappa_max,
        curvature_action=curvature_action,
        v_min=v_min,
        v_max=v_max,
        goal_dist_threshold=goal_dist_threshold,
        goal_heading_tolerance=goal_heading_tolerance,
        goal_end_wp_count=goal_end_wp_count,
        goal_end_wp_margin=goal_end_wp_margin,
        path_deviation_limit=path_deviation_limit,
        accel_max=accel_max,
        steer_rate_max=steer_rate_max,
        dt=dt,
        wheelbase=wheelbase,
        lidar_max_range=lidar_max_range,
        privileged_enabled=privileged_enabled,
        privileged_bins=privileged_bins,
        privileged_range=privileged_range,
        domain_rand_enabled=domain_rand_enabled,
        lidar_noise_std=lidar_noise_std,
        odom_drift_xy=odom_drift_xy,
        odom_drift_theta=odom_drift_theta,
        envs_panel_enabled=envs_panel_enabled,
        curriculum_enabled=curriculum_enabled,
        gear_lag_votes=gear_lag_votes,
        emit_reward_components=emit_reward_components,
        dynamics=dynamics,
        v_tau=v_tau,
        omega_tau=omega_tau,
        v_deadband=v_deadband,
        wheel_slip=wheel_slip,
    )


def _build_reward_config(reward_raw: dict) -> RewardConfig:
    """按主仓 TrainingConfig.from_config_dir 的取值顺序装配 RewardConfig。

    数值的**唯一权威是 reward.yaml**；YAML 缺某键时的兜底值一律取 dataclass
    的字段默认值（而不是又一堆手写字面量）。
    """
    reward_weights = dict(reward_raw.get("reward_weights", {}))
    rp = reward_raw.get("reward_params", {}) or {}
    _RD = RewardConfig()  # 兜底值来源

    dense_scale = float(rp.get("dense_scale", _RD.dense_scale))
    path_d_max = float(rp.get("path_d_max", _RD.path_d_max))
    cte_range = float(rp.get("cte_range", _RD.cte_range))
    cte_delta = float(rp.get("cte_delta", _RD.cte_delta))
    cte_dead_zone = float(rp.get("cte_dead_zone", _RD.cte_dead_zone))
    goal_time_bonus = float(rp.get("goal_time_bonus", _RD.goal_time_bonus))
    step_penalty = float(rp.get("step_penalty", _RD.step_penalty))
    obstacle_range = float(rp.get("obstacle_range", _RD.obstacle_range))
    collision_penalty = float(rp.get("collision_penalty", _RD.collision_penalty))
    # 撞墙附加罚：兜底值取 dataclass 字段，不写字面量（避免第三套默认值）。
    collision_surcharge = float(rp.get("collision_surcharge", _RD.collision_surcharge))
    if collision_surcharge < 0:
        raise ValueError(
            f"reward_params.collision_surcharge 必须 ≥ 0（0 = 关掉附加罚），"
            f"收到 {collision_surcharge}"
        )
    goal_bonus = float(rp.get("goal_bonus", _RD.goal_bonus))
    goal_heading_scale = float(rp.get("goal_heading_scale", _RD.goal_heading_scale))
    stuck_penalty = float(rp.get("stuck_penalty", _RD.stuck_penalty))
    timeout_penalty = float(rp.get("timeout_penalty", _RD.timeout_penalty))
    # 超时基项下限：兜底值取 dataclass 字段，不写字面量。
    timeout_min_penalty = float(rp.get("timeout_min_penalty", _RD.timeout_min_penalty))
    if not (timeout_min_penalty > 0.0):
        raise ValueError(
            f"reward_params.timeout_min_penalty 必须 > 0（0 等于关掉超时下限），"
            f"收到 {timeout_min_penalty}"
        )
    path_deviation_penalty = float(
        rp.get("path_deviation_penalty", _RD.path_deviation_penalty)
    )
    goal_proximity_weight = float(
        rp.get("goal_proximity_weight", _RD.goal_proximity_weight)
    )
    # 终点附近 obstacle 项门控（P2-14，2026-10-23 提进白名单 + reward.yaml）
    obstacle_near_goal_factor = float(
        rp.get("obstacle_near_goal_factor", _RD.obstacle_near_goal_factor)
    )
    obstacle_near_goal_range = float(
        rp.get("obstacle_near_goal_range", _RD.obstacle_near_goal_range)
    )
    if obstacle_near_goal_factor < 0:
        raise ValueError(
            f"reward_params.obstacle_near_goal_factor 必须 ≥ 0（0 = 终点附近整项清零），"
            f"收到 {obstacle_near_goal_factor}"
        )
    if obstacle_near_goal_range <= 0:
        raise ValueError(
            f"reward_params.obstacle_near_goal_range 必须 > 0，收到 {obstacle_near_goal_range}"
        )
    # 失败时刻补偿两参数（2026-10-05 接线）——默认值同样取 dataclass 字段
    failure_dense_cap = float(rp.get("failure_dense_cap", _RD.failure_dense_cap))
    failure_dense_refund_cap = float(
        rp.get("failure_dense_refund_cap", _RD.failure_dense_refund_cap)
    )
    if failure_dense_refund_cap < 0:
        raise ValueError(
            f"reward_params.failure_dense_refund_cap 必须 ≥ 0（0 = 不限），"
            f"收到 {failure_dense_refund_cap}"
        )
    failure_time_compensation = float(
        rp.get("failure_time_compensation", _RD.failure_time_compensation)
    )
    if failure_dense_cap < 0:
        raise ValueError(
            f"reward_params.failure_dense_cap 必须 ≥ 0，收到 {failure_dense_cap}"
        )
    if failure_time_compensation < 0:
        raise ValueError(
            f"reward_params.failure_time_compensation 必须 ≥ 0，收到 {failure_time_compensation}"
        )
    # stage_scale 的下限：必须落在 (0, 1] 内。
    # 0 ⇒ 低阶段失败惩罚整项消失；>1 ⇒ clip 上下界倒挂。
    stage_scale_floor = float(rp.get("stage_scale_floor", _RD.stage_scale_floor))
    if not (0.0 < stage_scale_floor <= 1.0):
        raise ValueError(
            f"reward_params.stage_scale_floor 必须满足 0 < x ≤ 1"
            f"（stage_scale 的下限），收到 {stage_scale_floor}"
        )
    # 速度感知障碍惩罚的幂次：只校验下界 ≥ 0（0 = 关闭）。
    # 上限**故意不强制**——>2 会过度放大高速罚分，只在注释里说明。
    obstacle_speed_power = float(
        rp.get("obstacle_speed_power", _RD.obstacle_speed_power)
    )
    if obstacle_speed_power < 0:
        raise ValueError(
            f"reward_params.obstacle_speed_power 必须 ≥ 0（0 = 关闭），"
            f"收到 {obstacle_speed_power}"
        )
    # 前进段倒退的专项定价：同样只校验下界 ≥ 0（0 = 关闭）。
    reverse_motion_penalty = float(
        rp.get("reverse_motion_penalty", _RD.reverse_motion_penalty)
    )
    if reverse_motion_penalty < 0:
        raise ValueError(
            f"reward_params.reverse_motion_penalty 必须 ≥ 0（0 = 关闭），"
            f"收到 {reverse_motion_penalty}"
        )

    return RewardConfig(
        reward_weights=dict(reward_weights),
        dense_scale=dense_scale,
        path_d_max=path_d_max,
        cte_range=cte_range,
        cte_delta=cte_delta,
        cte_dead_zone=cte_dead_zone,
        goal_time_bonus=goal_time_bonus,
        step_penalty=step_penalty,
        obstacle_range=obstacle_range,
        collision_penalty=collision_penalty,
        collision_surcharge=collision_surcharge,
        goal_bonus=goal_bonus,
        goal_heading_scale=goal_heading_scale,
        stuck_penalty=stuck_penalty,
        timeout_penalty=timeout_penalty,
        timeout_min_penalty=timeout_min_penalty,
        path_deviation_penalty=path_deviation_penalty,
        goal_proximity_weight=goal_proximity_weight,
        failure_dense_cap=failure_dense_cap,
        failure_dense_refund_cap=failure_dense_refund_cap,
        failure_time_compensation=failure_time_compensation,
        obstacle_near_goal_factor=obstacle_near_goal_factor,
        obstacle_near_goal_range=obstacle_near_goal_range,
        obstacle_speed_power=obstacle_speed_power,
        reverse_motion_penalty=reverse_motion_penalty,
        stage_scale_floor=stage_scale_floor,
    )


def load_default_config() -> tuple[EnvConfig, RewardConfig]:
    """读包内自带的 YAML（随 wheel 发布），返回 (EnvConfig, RewardConfig)。

    用 importlib.resources 定位 —— 与 CWD 无关，装成 wheel / zip 后同样可用。
    包内没有 train.yaml ⇒ 其专属字段取 dataclass 默认值（见模块 docstring）。
    """
    pkg = importlib.resources.files("chassis_sim.config")
    with importlib.resources.as_file(pkg) as config_dir:
        return from_config_dir(config_dir)
