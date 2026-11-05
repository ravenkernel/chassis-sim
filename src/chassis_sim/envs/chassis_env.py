"""车型参数化的环境核心（W4b）。

与原实现的**有意差异**（代码里也逐处注明）：
1. 默认桥是 chassis_sim.bridges.SimpleBridge（原实现要求外部注入 NumpySimBridge）；
   bridge=None 时按 cfg 自动构造。
2. 动作 → 物理量的映射由 Chassis.command() 负责，不再在 env 里写 δ/v 的
   if/else；阿克曼分支的公式与原实现逐位等价。
3. 不 import 宿主的训练配置（原实现从 training.config 取 EnvConfig），
   配置改用 chassis_sim.config.EnvConfig / RewardConfig（1.0.0 W1 起唯一家是
    config/schema.py）。
4. dt 形参真正生效（原实现的 dt 形参是死键，实际读 _ec.dt）。
5. MapGenerationError 按类名鸭子类型判定，不 import chassis_sim.planning
   （core 安装没有 numba）。
6. map_generator=None 时显式给内建直线场景（原实现传 None 让桥隐式兜底），
   两者几何完全一致（(1,5) → (9,5)，0.05 m 间隔）。
7. safety_gate 仅阿克曼可用（它夹的物理量是 (δ, v)）。
8. action 形状校验；mapped_action 宽度 = action_dim；观测 act 段宽度 =
   action_dim（差速 2 / 麦轮 3，见 _act_vector）。
9. 物理动作由 Chassis.command() 在 float64 里算出再交给桥；原实现把
    mapped_action 存成 float32 并用它积分。同一串动作下两侧状态会从 float32
    舍入位开始漂移，实测 100 步后观测 max|Δ| = 9.5e-07（float32 eps 的 8 倍），
    奖励 max|Δ| = 2.5e-08；无噪声的直线场景逐步 max|Δ| = 0。要位级一致，
    需让 chassis 包把 command() 的结果降到 float32（chassis/ 本次禁改）。
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chassis_sim.bridges import SensorProvider, SimpleBridge
from chassis_sim.chassis import Chassis, make_chassis
from chassis_sim.config import EnvConfig, RewardConfig
from chassis_sim.config.schema import _resolve, resolve_sensor_seed
from chassis_sim.obs import ObsSpec
from chassis_sim.obs.context import ContextBuilder
from chassis_sim.obs.spec import BoundContext
from chassis_sim.paths import gear_is_compliant, segment_is_reverse
from chassis_sim.reward import RewardCalculator
from chassis_sim.sensors import Imu, Lidar, WheelOdom

# 零速判据：|v| 小于它时无法由 ω 反解转角（与 controllers/base.py 的 V_EPS 同值）
V_EPS = 1e-6

# reset/step 的崩溃日志落盘位置：走系统临时目录而不是写死 /tmp（Windows 没有
# /tmp，写死的路径在那边会 FileNotFoundError，被下面的 except 吞掉 ⇒ 诊断丢失）。
_CRASH_LOG = Path(tempfile.gettempdir()) / "ackermann_worker_crash.log"

# 字符串车型 → make_chassis 的参数白名单：键名必须同时存在于 EnvConfig 与对应
# Chassis 的构造签名（差速/麦轮不收 v_min/delta_max/kappa_max 等）。
_DEFAULT_CHASSIS_PARAMS: dict[str, tuple[str, ...]] = {
    "ackermann": (
        "wheelbase",
        "delta_max",
        "v_min",
        "v_max",
        "curvature_action",
        "kappa_max",
    ),
    "diff_drive": ("wheelbase", "v_max"),
    "mecanum": ("v_max",),
}

# 默认世界边界（与 config/env.yaml 的 world 段、SimpleBridge 默认值一致）
_DEFAULT_WORLD_BOUNDS = (0.0, 10.0, 0.0, 10.0)

#: 默认场景重掷参数住在 chassis_sim.scenarios.reset_options.roll_obstacles，
#: 与 evaluate/benchmark 的逐回合工厂共用同一份（勿在此处再定义一套）。

#: matplotlib 的非交互后端：human 渲染在这些后端上只画不刷屏。
_NON_INTERACTIVE_BACKENDS = frozenset(
    {"agg", "pdf", "ps", "svg", "template", "cairo", "pgf"}
)


def _resolve_chassis(chassis, cfg: EnvConfig) -> Chassis:
    """车型名 → make_chassis(名, **cfg 同名参数)；Chassis 实例 → 原样返回。"""
    if isinstance(chassis, str):
        params = _DEFAULT_CHASSIS_PARAMS.get(chassis)
        if params is None:
            raise ValueError(
                f"未知车型 {chassis!r}；可用 {sorted(_DEFAULT_CHASSIS_PARAMS)}，"
                f"或直接传 Chassis 实例"
            )
        return make_chassis(chassis, **{k: getattr(cfg, k) for k in params})
    if not isinstance(chassis, Chassis):
        raise TypeError(
            f"chassis 必须是车型名或 Chassis 实例，收到 {type(chassis).__name__}"
        )
    return chassis


def _is_map_generation_error(exc: BaseException) -> bool:
    """鸭子类型判定 MapGenerationError：不 import planning（core 无 numba）。"""
    return any(cls.__name__ == "MapGenerationError" for cls in type(exc).__mro__)


def _straight_scene() -> tuple[list, np.ndarray, float, float]:
    """内建直线兜底场景：start (1,5) → goal (9,5)，0.05 m 间隔，航向 0。

    路径是 **(N, 3) [x, y, 鼻向]**：env 的 CTE / 档位判定读 ``path_body[:, 2]``，
    桥侧的隐式兜底同样带 θ 列（用
    ``column_stack([path_x, path_y, theta])``），列数必须对齐。
    """
    n_pts = max(2, int(np.ceil(8.0 / 0.05)) + 1)
    path = np.zeros((n_pts, 3), dtype=np.float64)
    path[:, 0] = np.linspace(1.0, 9.0, n_pts)
    path[:, 1] = 5.0
    path[:, 2] = 0.0
    return [], path, 0.0, 0.0


class GearVoteLag:
    """档位投票滞后状态机 (2026-10-06)：需要连续 N 步同向才切换档位判定。

    **为什么需要**（实测，真 stage3/stage4 地图，老师闭环 40 集/臂）：
    env 的 `path_dir` 由"全局最近点 + 中心差分运动方向"逐点算出，在 cusp、
    自接近区（折返/迷宫平行走廊）会**每步抖动**：raw path_dir 的**每步翻符号率
    50.1%(s3) / 52.5%(s4)**，恒定段长度分布里 4281/4757 段只有 1 步 ⇒ 这个信号
    接近白噪声。而"档位闸门"（`gear_is_compliant`）与倒车段偏航加权都直接消费
    这个 raw 值 ⇒ **奖励在惩罚噪声**：一辆正确倒车的车也被随机判违规、
    `arc_progress` 被清零。同一批**逐位相同**的轨迹只换闸门判定，实测：
        合规推进分总和  raw 闸门 **8.76** → 滞后闸门 **205.98**（stage3, 40 集）
                        raw 闸门 **33.92** → 滞后闸门 **159.94**（stage4, 40 集）
    即旧闸门吃掉了 96%(s3) / 79%(s4) 的合规推进分，而且是随机吃的。
    教师侧（`/tmp/teacher/tlaw.py`）加 3 步滞后后倒车段 |CTE| 0.135/0.144
    → **0.062/0.046**、反向步推进率 28%/21% → **54%/59%**、成功率
    42.5%/27.5% → **50.0%/40.0%**（`/tmp/w5a/out_lag_ab*.json`）。

    **不进入观测**：滞后只作用于 env 内部的档位判定（闸门 + 偏航权重 +
    info 诊断），策略侧看不到 path_dir，档位技能仍要自己从路径几何学。

    投票语义与仓库自带 `PurePursuitController`（`hysteresis=3`）和
    教师律 `TeacherLaw.filt_dir` 逐位一致：票数在 [-N, N] 饱和，
    只有票数触边才切换稳定档位；过渡期**保持上一稳定档位**。
    `votes <= 1` 时完全等价于旧行为（raw 直通）。
    """

    def __init__(self, votes: int = 3):
        self.votes = max(1, int(votes))
        self.reset()

    def reset(self) -> None:
        self._counter = 0
        self._stable = None

    def update(self, raw_dir: float) -> float:
        """喂入本步 raw path_dir，返回滞后后的稳定档位 (±1)。"""
        if self.votes <= 1:
            self._stable = raw_dir < 0
            return -1.0 if self._stable else 1.0
        if raw_dir < 0:
            self._counter = min(self._counter + 1, self.votes)
        else:
            self._counter = max(self._counter - 1, -self.votes)
        if self._counter >= self.votes:
            self._stable = True
        elif self._counter <= -self.votes:
            self._stable = False
        if self._stable is None:  # 首步：直接采纳 raw
            self._stable = raw_dir < 0
        return -1.0 if self._stable else 1.0


# ══════════════════════════════════════════════════════════════════════
# RewardCalculator 参数签名校验 (P2-18)
#
# 病因：AckermannEnv.__init__ 把 12 个参数显式转发给 RewardCalculator，而后者
# 的签名里**没有**它们 —— 全被 **kwargs 吞掉再丢弃（只有构造体里
# kwargs.get('x') 显式读过的那几个才算真被消费）。env 侧看起来是"活接线"，
# 实际改 YAML / 传参都不生效，且**没有任何报错**（本项目反复踩的静默丢弃坑）。
#
# 修法：不改 reward.py（本轮 X4 的边界），在 env 侧按**真实签名**校验，
# 凡是不被接受的键打一次 [WARN][env]（strict 模式直接 raise）。
# ══════════════════════════════════════════════════════════════════════

# 本进程已告警过的键集合（"打一次"而不是每个 env 实例打一次：单进程评估会
# 反复建 env，训练是 16 个 spawn worker —— 每进程一条就够，不刷屏）。
_WARNED_REWARD_KWARGS: set[frozenset[str]] = set()


@functools.lru_cache(maxsize=1)
def reward_ctor_accepted_names() -> frozenset[str]:
    """RewardCalculator.__init__ **真正会消费**的参数名集合。

    两类来源：
      ① 签名里的具名参数（POSITIONAL_OR_KEYWORD / KEYWORD_ONLY）；
      ② 构造函数体里 kwargs.get('<name>' 显式读取的键 —— 这些虽然被
         **kwargs 吞下，但**确实被消费**（obstacle_near_goal_factor /
         obstacle_near_goal_range 走的就是这条路）。

    其余键（能传进去、签名与函数体都没用到）会被静默丢弃 ⇒ 属于"假接线"。
    用源码扫描而不是手工白名单：reward.py 以后新增 kwargs.get 键时自动被
    认作"已接受"，不会误报。
    """
    from chassis_sim.reward import RewardCalculator

    sig = inspect.signature(RewardCalculator.__init__)
    accepted = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    try:
        src = inspect.getsource(RewardCalculator.__init__)
    except (OSError, TypeError):  # 源码不可得（编译/打包）⇒ 退化为仅签名
        return frozenset(accepted)
    # 只认 kwargs.get('x') / kwargs["x"] 两种写法（本仓库用的是前者）
    accepted.update(re.findall(r"""kwargs\.get\(\s*['"](\w+)['"]""", src))
    accepted.update(re.findall(r"""kwargs\[\s*['"](\w+)['"]\s*\]""", src))
    return frozenset(accepted)


def reward_kwargs_not_accepted(kwargs: Mapping[str, Any]) -> list[str]:
    """返回 kwargs 里 RewardCalculator 不会消费的键（保持传入顺序）。"""
    ok = reward_ctor_accepted_names()
    return [k for k in kwargs if k not in ok]


class ChassisSimEnv(gym.Env):
    """车型参数化的底盘仿真环境。

    <<chassis>> 是车型名（"ackermann" / "diff_drive" / "mecanum"）或 Chassis
    实例。动作空间、物理映射、观测 act 段宽度都由底盘派生；其余语义（观测
    组装顺序、奖励、终止判定、info 键）与原实现一致。

    Action: Box(action_dim, [-1,1], float32)，各轴语义由底盘定义：
      * ackermann: [转向标度, 速度标度]
      * diff_drive: [速度标度, 横摆标度]
      * mecanum: [速度标度, 横移标度, 横摆标度]

    三个构造器开关各管一层：dynamics= 只是**覆盖** cfg 选中的动力学档位
    （"none"/"limits"/"wheel"，None = 按 cfg）；sensors= 是**装配**一套传感器
    （由 env 自己造的桥持有，每支一条独立随机流，seed 由 sensor_seed= 给；
    读数只经 info 曝光，除非 obs_spec 里的 lidar 段显式引用它）；obs_spec=
    决定观测怎么拼（None ⇒ ObsSpec.default()，即
    odom+path+goal+act 的 128 维最小集；v0.1.4 的 494 维要显式传 9 段 ObsSpec，
    其中 lidar 段引用已装配的两支 Lidar 传感器名）。

    两个种子形参（1.1.0 D）分工不同：seed= 是**运行**种子，env 自造桥时传给
    SimpleBridge(seed=...)，域随机化噪声（LiDAR 加性噪声 + 里程计漂移）因此走桥
    自持的 np.random.default_rng(seed) ⇒ 同 seed 逐位可复现；sensor_seed= 只管
    传感器随机流，None ⇒ 跟随 seed，两者都没有 ⇒ 0（推导规则见
    config.schema.resolve_sensor_seed）。seed=None（默认）时域随机化仍吃全局
    np.random，即 1.0.x 的既有行为、数值逐位不变。

    **注入桥时不接管随机源**：env 只把 seed 传给**自己造的** SimpleBridge；
    bridge= 注入的桥的随机性由桥自己负责（它自己的 seed=、或它自己的全局状态），
    env 既不读也不写它的 Generator / np.random —— 否则同一个桥被两个 env 共用时
    会互相打乱随机流。sensors= 与 bridge= 互斥，注入桥上也没有任何种子可传。
    """

    # 类级默认（gymnasium 读 ClassVar）；实例 metadata 在 __init__ 里补 render_fps，
    # render_fps 由 dt 推，不能在类级写死。
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    # ── 有界追赶参数（2026-10-06：修 _target_idx 永久冻结）────────────────
    # 详见 `_advance_target_idx` 的 docstring。
    CATCHUP_HEADING_TOL = 1.2  # 候选点鼻向与车头朝向的最大夹角 (rad)
    CATCHUP_MARGIN = 0.25  # 自适应半径余量 (m)
    CATCHUP_RADIUS_MAX = 1.25  # 自适应半径上限 (m)
    CATCHUP_ARC_K = 5.0  # 单步弧长预算 = max|v|·dt·K
    MAP_GEN_RETRIES = 8  # 地图生成重试上限（P2-20：原 100，见 reset）

    def __init__(
        self,
        chassis: Any = None,
        *,
        bridge: SensorProvider | None = None,
        sensors: Sequence[Lidar | Imu | WheelOdom] | None = None,
        dynamics: str | None = None,
        cfg: EnvConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        max_steps: int | None = None,
        map_generator=None,
        safety_gate=None,
        domain_rand: bool = False,
        env_index: int = 0,
        status_board=None,
        dt: float | None = None,
        path_lookahead: float | None = None,
        max_path_points: int | None = None,
        obs_spec: ObsSpec | None = None,
        sensor_seed: int | None = None,
        seed: int | None = None,
        path_power: float | None = None,
        extrapolate_path: bool | None = None,
        path_fill_mode: str | None = None,
        path_lookahead_floor: float | None = None,
        reward_weights: dict | None = None,
        stage_index: int = -1,
        reward_kwargs_extra: dict | None = None,
        strict_reward_kwargs: bool = False,
        emit_reward_components: bool | None = None,
        world_bounds: tuple[float, float, float, float] | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        _ec = cfg if cfg is not None else EnvConfig()
        _rc = reward_cfg if reward_cfg is not None else RewardConfig()
        _rw = reward_weights if reward_weights is not None else _rc.reward_weights

        # ── 车型（W4b）────────────────────────────────────────────────
        # cfg 只是车型参数的**输入口**：构造完成后一切物理口径以
        # chassis.describe() 为准，避免 env 与底盘各存一份参数而漂移。
        # 形参 None = 没给 ⇒ 逐级回落 cfg 字段（api.make 与本构造器共用 _resolve）
        self.chassis = _resolve_chassis(
            _resolve(chassis, _ec.chassis, "ackermann"), _ec
        )
        _desc = self.chassis.describe()
        self.chassis_name = str(_desc["name"])
        self.action_dim = int(self.chassis.action_dim)
        _lim = _desc["limits"]
        _par = _desc["params"]
        self.v_min = float(_lim["v_min"])
        self.v_max = float(_lim["v_max"])
        self.omega_max = float(_lim["omega_max"])
        self.vy_max = float(_lim.get("vy_max") or 0.0)
        self.wheelbase = float(_par.get("wheelbase") or _ec.wheelbase)
        self.kappa_max = float(_par.get("kappa_max") or _ec.kappa_max)
        # delta_max 只对阿克曼是物理量；差速/麦轮在 describe() 里给 None，
        # 这里退回 cfg 值作为「转向维标度」的除数（见 _physical_action）。
        _dm = _par.get("delta_max")
        self.delta_max = float(_dm) if _dm is not None else float(_ec.delta_max)
        self.curvature_action = bool(_par.get("curvature_action"))
        if safety_gate is not None and self.chassis_name != "ackermann":
            raise ValueError(
                "safety_gate 只对阿克曼定义（它夹的是物理 (δ, v)，见 "
                f"envs.safety_gate）；{self.chassis_name} 请传 None"
            )

        self.map_generator = map_generator
        # 课程阶段索引（B6）：-1 = 未知。作为 reward stage_scale 的权威来源，
        # 即便 map_generator 为 None（无课程/评估环境）也能正确缩放终止惩罚。
        self._stage_index = int(stage_index) if stage_index is not None else -1
        self.max_steps = int(_resolve(max_steps, _ec.max_steps, 1000))
        self.goal_threshold = _ec.goal_dist_threshold
        self._goal_end_wp_count = _ec.goal_end_wp_count
        self.path_deviation_limit = _ec.path_deviation_limit
        self._priv_enabled = _ec.privileged_enabled
        self._max_path_points = int(_resolve(max_path_points, _ec.max_path_points, 30))
        self._path_power = float(_resolve(path_power, _ec.path_power, 2.0))
        self._obstacles: list[dict] = []
        # 当前生效的动态障碍规格（空 = 无）；切回静态场景时用于显式清空。
        self._dynamic_obstacles: list[dict] = []
        # dt 真生效（原实现的 dt 形参是死键，实际读 _ec.dt）：桥与
        # step 必须共用同一个 dt，所以先于桥构造赋值。
        # 三级优先级保留：dt kwarg > cfg.dt > dataclass 默认（cfg.dt 恒有值）
        self.dt = float(_resolve(dt, _ec.dt))
        self._last_delta = 0.0  # 零速时 δ 无法由 ω 反解，沿用上一拍

        # ── 动力学三级开关（2026-11-04，语义见 envs/dynamics.py）──────────
        # 档位来源：构造器 dynamics= 给值就**覆盖** cfg 的档位（EnvConfig.dynamics，
        # 即 env.yaml 的 robot.dynamics.level）；None（默认）= 完全按 cfg 走，行为与
        # 加入本参数前逐位相同。其余动力学参数（accel_max / v_tau / ...）始终取 cfg。
        # 默认 "none" 分支**不 import** dynamics.py：模块不被加载、不被调用、
        # 不消耗随机数 ⇒ 与加入本开关前逐位相同（_accel_max/_steer_rate_max 原样
        # 透传，0 仍是 0，限幅通路一个点都不进）。
        # 依赖 self.dt（一阶滞后的每步收敛比例），所以在 dt 之后构造。
        _src_level = _ec.dynamics if dynamics is None else dynamics
        self._dynamics_level = str(_src_level).strip().lower()
        if self._dynamics_level == "none":
            if dynamics is None:
                # cfg 路径：限幅值原样透传 —— 这是历史行为（档位默认就是 none，
                # 而 EnvConfig(accel_max>0) 照样限幅，见 tests/test_envs_diff_drive_limits.py），
                # 改它就不是"逐位不变"了。
                self._accel_max = float(_ec.accel_max)
                self._steer_rate_max = float(_ec.steer_rate_max)
            else:
                # 构造器显式选 none ⇒ 以 kwarg 为准：none 档是纯运动学、不看限幅，
                # cfg 里的 accel_max / steer_rate_max 一律不生效（置 0 后
                # _apply_limits_and_gate 的限幅分支整段不进；SafetyGate 仍按自己的
                # 开关走，它不属动力学档）。这就是"档位与限幅冲突时 kwarg 赢"。
                self._accel_max = 0.0
                self._steer_rate_max = 0.0
            self._dynamics = None
        else:
            from chassis_sim.envs.dynamics import (  # 局部 import：none 档不加载
                ActuatorDynamics,
                resolve_level,
                resolve_limits,
            )

            self._dynamics_level = resolve_level(self._dynamics_level)  # 非法值在此抛
            self._accel_max, self._steer_rate_max = resolve_limits(
                self._dynamics_level, _ec.accel_max, _ec.steer_rate_max
            )
            self._dynamics = (
                ActuatorDynamics(
                    dt=self.dt,
                    v_tau=_ec.v_tau,
                    omega_tau=_ec.omega_tau,
                    v_deadband=_ec.v_deadband,
                    wheel_slip=_ec.wheel_slip,
                )
                if self._dynamics_level == "wheel"
                else None
            )
        # 实际生效值（info["dynamics"] / describe_dynamics()）：选 limits/wheel 而
        # YAML 里写 0 时这里会显示**解析后的**默认值，不是 0。
        self._dynamics_info: dict = {
            "level": self._dynamics_level,
            "accel_max": float(self._accel_max),
            "steer_rate_max": float(self._steer_rate_max),
        }
        if self._dynamics is not None:
            self._dynamics_info.update(self._dynamics.describe())
        if dynamics is not None:
            # 只在构造器给档位时多这个键：默认路径的 describe_dynamics() 与加入
            # 本参数前**逐键逐值相同**（tests/test_envs_dynamics.py 的 TestNoneLevel
            # 对 dict 做等值比较）。有它就能看出"level 是构造器参数指定的"。
            self._dynamics_info["level_source"] = "constructor"

        # ── 桥：默认 SimpleBridge；注入的桥必须共用同一台底盘 ──────────
        # env 是否**自己造了**桥（只有 bridge=None 才会自造 SimpleBridge）。这是
        # 一次性的构造期事实，必须在这里记下：走到 self.bridge = bridge 之后，
        # "自造"与"注入"已无从分辨，任何事后重新推断都只能退化成猜。默认场景的
        # 接管判据消费它（见 _default_scenario_active）。
        self._bridge_owned = bridge is None
        _bridge_chassis = getattr(bridge, "chassis", None)
        if _bridge_chassis is not None and _bridge_chassis is not self.chassis:
            raise ValueError(
                "注入的 bridge 用的是另一台 Chassis 实例：动作映射与物理必须"
                "同源，请传同一个实例（如 SimpleBridge(self.chassis, ...)）"
            )
        # env 自造桥时记下传感器声明：只为给 obs_spec 的 lidar 段补 n_rays
        # （注入 bridge= 的路径上 sensors= 会被下面拒绝，这里存的是空元组）
        sensors = _resolve(sensors, _ec.sensors, None)
        self._sensor_specs: tuple[Any, ...] = () if sensors is None else tuple(sensors)
        # 每个 lidar 名最后一次**有效**扫描（S3b 零阶保持）：丢帧 / latency 预热
        # 期用它顶上，绝不谎报"四周空旷"。只存真实读数，reset 时清空。
        self._lidar_last_scan: dict[str, np.ndarray] = {}
        # ── 随机种子解析（1.1.0 D）───────────────────────────────────────
        # seed         = 运行种子：域随机化噪声（LiDAR 加性噪声 + 里程计漂移）的
        #                随机源，非 None 时由 env 自造的桥自持一条 Generator。
        # sensor_seed  = 传感器随机流种子，None ⇒ 跟随 seed（唯一推导点见
        #                config.schema.resolve_sensor_seed），两者都没有 ⇒ 0。
        # 两者都走既有的三层解析（kwargs > EnvConfig 字段 > 库默认）。
        _seed = _resolve(seed, _ec.seed, None)
        self.seed: int | None = None if _seed is None else int(_seed)
        self.sensor_seed: int = resolve_sensor_seed(
            _resolve(sensor_seed, _ec.sensor_seed, None), self.seed
        )
        # ── 内建默认场景（3.0.0 W1）─────────────────────────────────────
        # 调用方没给场景时不再退化成固定直线：env 按 EnvConfig.default_scenario
        # 接一张自带障碍的场景，并允许重掷地图（只在 options 为 None **且**桥由
        # env 自造时生效，见 _default_scenario_active）。default_scenario=None ⇒
        # 回到 2.x 的直线。
        self._default_scenario_name = getattr(_ec, "default_scenario", None)
        self._default_scenario_obj = None
        self._default_roll_index = 0
        # world_bounds 无论桥是不是注入的都解析并留存：观测界（goal/path 段）要用
        # 它推导"车与路径都在这个矩形里"，这是唯一一条几何前提。注入桥时这个值
        # 只用于推导，绝不回写桥（桥自己的边界由桥负责）。
        _wbounds = tuple(
            float(_x)
            for _x in _resolve(world_bounds, _ec.world_bounds, _DEFAULT_WORLD_BOUNDS)
        )
        self._world_bounds = _wbounds
        if bridge is None:
            bridge = SimpleBridge(
                self.chassis,
                dt=self.dt,
                world_bounds=_wbounds,
                max_range=float(_ec.lidar_max_range),
                domain_rand=bool(domain_rand),
                # 只有 env 自造的桥才拿到 seed；注入桥的随机性由桥自己负责（见
                # 类 docstring）—— 这里绝不改别人的 np.random 或 Generator。
                seed=self.seed,
                lidar_noise_std=float(_ec.lidar_noise_std),
                odom_drift_xy=float(_ec.odom_drift_xy),
                odom_drift_theta=float(_ec.odom_drift_theta),
                goal_dist_threshold=float(_ec.goal_dist_threshold),
                goal_end_wp_count=int(_ec.goal_end_wp_count),
                goal_end_wp_margin=int(_ec.goal_end_wp_margin),
                sensors=sensors,
                sensor_seed=self.sensor_seed,
                # 积分/碰撞精度旋钮（2026-11-05）：只转发，不在这里解释语义。
                # 注入桥时**不回写**（桥自己拥有它的积分器与碰撞判据）。
                integration_substeps=int(_ec.integration_substeps),
                ccd=bool(_ec.ccd),
            )
        elif sensors is not None:
            # sensors= 与注入 bridge= **互斥**（这里选择抛错而不是"事后转交"）：
            # 传感器套件由 SimpleBridge.__init__ 一次装配并接管 reset()/step() 的更新
            # 时序（bridges/simple_bridge.py:197-205、:274-275、:455），bridges/** 是
            # 本次禁改区且没有 post-hoc 挂载接口 ⇒ env 无法在不复制桥内部状态机的前提下
            # 把传感器装进**已构造**的桥；而"悄悄忽略"会让用户以为传感器开了却没开
            # （更难查）。已自带 sensors 的桥同理：再传就是两套状态互相覆盖。
            raise ValueError(
                "sensors= 只在 env 自己造桥（bridge=None）时生效：已注入的 "
                f"{type(bridge).__name__} 不接受事后挂载传感器（SimpleBridge 只在 "
                "__init__ 里装配 SensorSuite，且没有 set_sensors）。二者选一："
                "去掉 bridge=（让 env 用 sensors= 造桥），或去掉 sensors="
                "（在构造桥时自行传 sensors=）"
            )
        self.bridge = bridge
        self.domain_rand = domain_rand
        self._safety_gate = safety_gate  # 训练时为 None，评估时为 SafetyGate

        # 用于 Asymmetric Actor-Critic 的 Privileged encoder
        if _ec.privileged_enabled:
            from chassis_sim.envs.privileged import PrivilegedEncoder

            # 提取世界边界供特权编码器使用，修复 Wall Ghost Bug——
            # 否则 Critic 看不到墙壁，Actor 看到墙壁会导致 Q 值错位
            _wb = None
            if bridge is not None:
                try:
                    _wb = (
                        float(bridge._world_xmin),
                        float(bridge._world_xmax),
                        float(bridge._world_ymin),
                        float(bridge._world_ymax),
                    )
                except AttributeError:
                    pass
            self._priv_encoder = PrivilegedEncoder(
                n_bins=_ec.privileged_bins,
                max_range=_ec.privileged_range,
                world_bounds=_wb,
            )
        else:
            self._priv_encoder = None

        # Action: tianshou 输出 tanh→[-1,1]；物理映射由底盘负责
        self.action_space = self.chassis.action_space()

        # ── Reward ─────────────────────────────────────────────────────
        # 【P2-18】这些键按**显式清单**转发给 RewardCalculator（没有 **kwargs
        # 兜底），因此必须保证每个键都真的在奖励计算器里被消费；否则就是
        # "env 侧看起来活着、实际被 **kwargs 吞掉"的假接线。构造前用
        # reward_kwargs_not_accepted() 按**真实签名 + kwargs.get 扫描**校验。
        # 0.1.1：删掉 11 个从未被 RewardCalculator 消费的键（原 WARN 的 12 个减去
        # path_d_max —— 它另有观测侧的显式通路，见下方 P2-19）：
        # reverse_step_penalty / alive_bonus / curvature_scale / heading_alpha
        # 来自旧 RewardConfig；speed_gate_threshold / path_d_scale /
        # progress_gate_threshold / progress_gate_mode 是本文件里的字面量；
        # obs_danger / obs_safe / stop_threshold 是 EnvConfig 的字段（字段已删）。
        # 护栏本身保留：将来再加死键，这里仍会 warn（strict 下 raise）。
        _rw_kwargs: dict[str, Any] = {
            "weights": _rw,
            "obstacle_range": _rc.obstacle_range,
            "collision_penalty": _rc.collision_penalty,
            "goal_bonus": _rc.goal_bonus,
            "goal_heading_scale": _rc.goal_heading_scale,
            "dense_scale": _rc.dense_scale,
            "stuck_penalty": _rc.stuck_penalty,
            "timeout_penalty": _rc.timeout_penalty,
            "path_deviation_penalty": _rc.path_deviation_penalty,
            "v_min": _ec.v_min,
            "goal_time_bonus": _rc.goal_time_bonus,
            "cte_range": _rc.cte_range,
            "step_penalty": _rc.step_penalty,
            "cte_delta": _rc.cte_delta,
            # 失败时刻补偿三参数（2026-10-05 接线）：此前 RewardConfig 白名单没有
            # 这些键 ⇒ reward.yaml 里写了也被静默丢弃，只能从构造函数覆盖。
            # getattr 兜底：兼容没有这些字段的旧 RewardConfig 对象。
            # 兜底值必须与 RewardConfig 的字段默认逐位一致（现为 0.0 = 关闭；
            # 2026-10-06 X4 把 RewardConfig 对齐到 reward.yaml 的 0.0，这里原来
            # 还写着 0.12 ⇒ 会成为"第五套值"：只在传入缺该字段的旧 RewardConfig
            # 时生效，与三向一致口径不符）。
            "failure_dense_cap": getattr(_rc, "failure_dense_cap", 0.0),
            "failure_dense_refund_cap": getattr(_rc, "failure_dense_refund_cap", 0.0),
            "failure_time_compensation": getattr(_rc, "failure_time_compensation", 0.0),
            # near_goal 内的 obstacle 衰减（2026-10-06 接线）：RewardCalculator 从
            # **kwargs 读这两个键 ⇒ 必须在 env 侧显式转发，否则 reward.yaml 里
            # 配了也等于没配。默认值与 envs/reward.py 的 kwargs.get 默认逐位一致
            # （factor=0.0 ⇒ near_goal 内 obstacle 项仍按旧行为硬清零）。
            # 这两个键**会被** reward.py 的 kwargs.get 消费，所以不算"未接受参数"
            # （见 reward_ctor_accepted_names() 的第 ② 类来源）。
            "obstacle_near_goal_factor": getattr(_rc, "obstacle_near_goal_factor", 0.0),
            "obstacle_near_goal_range": getattr(_rc, "obstacle_near_goal_range", 2.0),
            # 速度感知障碍惩罚（2026-10-23 接线）：**具名参数**，不转发 ⇒ reward.yaml
            # 配了也等于没配。兜底值与 RewardConfig / _SRewardParams / reward.yaml /
            # RewardCalculator 签名逐位一致（0.0 = 关闭 ⇒ 行为逐位不变）。
            # 因子用的 v / v_max 都不需要在 env 侧额外传：compute() 的形参里已有，
            # 且 _step_impl 每步都传 v=实际执行车速、v_max=self.v_max。
            "obstacle_speed_power": getattr(_rc, "obstacle_speed_power", 0.0),
            # 前进段倒退的专项定价（2026-10-29 接线）：同样是**具名参数**，不转发
            # ⇒ reward.yaml 配了也等于没配。判据用的 path_dir 与 v 是 compute() 的
            # 既有形参（_step_impl 每步传 v=实际执行车速、path_dir=滞后稳定档位）。
            "reverse_motion_penalty": getattr(_rc, "reverse_motion_penalty", 0.0),
            # stage_scale 的下限（2026-10-23 接线）：这是 RewardCalculator 的**具名参数**
            # （不是 kwargs.get 键），不转发 ⇒ reward.yaml 配了也等于没配。
            # 兜底值与 RewardConfig / reward.py 签名 / reward.yaml 逐位一致（0.2），
            # 只在传入缺该字段的旧 RewardConfig 对象时生效。
            "stage_scale_floor": getattr(_rc, "stage_scale_floor", 0.2),
            # 「失败经济学」三件套（2026-10-23 接线）：两个都是 RewardCalculator 的**具名
            # 参数**，不转发 ⇒ reward.yaml 配了也等于没配（本项目记录在案的
            # 「YAML 旋钮静默失效」）。兜底值与 RewardConfig / reward.py 签名 /
            # reward.yaml 三向一致（60.0 / 30.0）；getattr 兜底只为兼容旧 RewardConfig。
            "collision_surcharge": getattr(_rc, "collision_surcharge", 60.0),
            "timeout_min_penalty": getattr(_rc, "timeout_min_penalty", 30.0),
        }
        if reward_kwargs_extra:
            _rw_kwargs.update(reward_kwargs_extra)
        _rejected = reward_kwargs_not_accepted(_rw_kwargs)
        if _rejected:
            _msg = (
                f"[WARN][env] {len(_rejected)} 个奖励参数不会被 RewardCalculator 消费"
                f"（签名与构造体的 kwargs.get 都没用到），传进来等于没接线: "
                f"{', '.join(_rejected)}"
            )
            if strict_reward_kwargs:
                raise ValueError(_msg)
            _key = frozenset(_rejected)
            if _key not in _WARNED_REWARD_KWARGS:  # 每进程只打一次，避免刷屏
                _WARNED_REWARD_KWARGS.add(_key)
                print(_msg, file=sys.stderr, flush=True)
        self.reward_calc = RewardCalculator(**_rw_kwargs)
        # 本实例实际被丢弃的奖励参数名（供测试/诊断读取，不改 info schema）
        self.reward_kwargs_rejected: tuple[str, ...] = tuple(_rejected)

        # ── reward_components 开关（P2-22）──────────────────────────────
        # 一个 flag 一个语义：只关**训练路径零消费者的 reward_components**
        # （7~15 键 dict，实测 1711 B/条），mapped_action 永远保留。
        #
        # 为什么不是两个都关：mapped_action 有真消费者 —— EnvStatsWrapper 用
        # 它算 delta_mean/delta_std/v_mean/v_std → TensorBoard 的
        # action/delta_mean、action/v_mean + EnvStatusBoard 环缓冲 + TUI 面板，
        # 单位是 rad 与 m/s；缺键时它回退到归一化动作 ⇒ 数值不崩但**单位变**
        # （为了 302 B 换诊断语义变味不划算）。
        # 为什么 reward_components 可以关：EnvStatsWrapper 只在回合末做
        # _average_reward_comps，而 Trainer._log_episodes / CSVWriter.write_episode
        # / TrainingLogger.log_episode 都不读它 ⇒ 训练主路径零消费者。
        # 真消费者只有 scripts/eval.py 与若干测试（它们用默认 True）。
        #
        # 内存估算（真 stage3 环境 200 步实测，/tmp/x6/measure_after.py）：
        #   False ⇒ 1539 B/条 vs 改前 2884 B/条（−47%），1e6 条 ≈ −1.28 GB；
        #   参照：obs 494×float32 才 1976 B/条。
        # 生产训练建议置 False（需 S3 把 EnvConfig/train.yaml/env_factory 接上）。
        # cfg 字段与 kwarg 同一解析链：cfg=False 曾经是死键（永远被形参默认 True 盖住）
        self._emit_reward_components = bool(
            _resolve(emit_reward_components, _ec.emit_reward_components, True)
        )

        # ContextBuilder — raw 传感器字典 → ObsContext（编码器已移入 Actor）
        # 【P2-19 接线】path_d_max 是 CTE 的**观测归一化除数**（err 分量第 0 维
        # cte/path_d_max，clip 到 ±1），从 RewardConfig 转发；getattr 兜底兼容
        # 没有该字段的旧 RewardConfig。0.1.1：它**不是**奖励参数（RewardCalculator
        # 从不消费）⇒ 只从 _rw_kwargs 摘掉，字段 / YAML 键 / 本行一律保留。
        # lookahead 存一份：ContextBuilder 用它截取窗口，观测界（path 段的 ρ）
        # 也要用它推导上界，两处必须是同一个数。
        self._path_lookahead = float(_resolve(path_lookahead, _ec.path_lookahead, 20.0))
        self._ctx_builder = ContextBuilder(
            max_path_points=self._max_path_points,
            path_power=self._path_power,
            lookahead=self._path_lookahead,
            v_max=self.v_max,
            v_min=self.v_min,
            delta_max=self.delta_max,
            path_d_max=getattr(_rc, "path_d_max", 4.0),
            extrapolate_path=bool(
                _resolve(extrapolate_path, _ec.extrapolate_path, True)
            ),
            path_fill_mode=str(
                _resolve(path_fill_mode, _ec.path_fill_mode, "extrapolate")
            ),
            path_lookahead_floor=float(
                _resolve(path_lookahead_floor, _ec.path_lookahead_floor, 1.0)
            ),
        )

        # Observation: 段布局完全由 ObsSpec 决定（0.2.0），Critic 的 privileged
        # 尾部仍追加在整条向量最末（因此不影响 spec 内部的切片）。
        priv_dim = _ec.privileged_bins if _ec.privileged_enabled else 0
        if obs_spec is None:
            obs_spec = _ec.obs_spec
        if obs_spec is None:
            # 默认最小集 odom+path+goal+act ⇒ 128 维（麦轮 act 3 维 ⇒ 129）。
            obs_spec = ObsSpec.default(
                action_dim=self.action_dim, path_points=self._max_path_points
            )
        self.obs_spec = obs_spec
        # resolved() 一次钉死各段宽度：lidar 段缺射线数在这里就报错，而不是等
        # 第一个 step 拼观测时才炸（那时错误离原因已经很远）。
        self._obs_resolved = self.obs_spec.resolved(self._sensor_meta())
        self._obs_slices = self._obs_resolved.describe()
        self._base_obs_dim = self._obs_resolved.total_dim
        self._full_obs_dim = self._base_obs_dim + priv_dim
        _act_sl = self._obs_slices.get("act")
        if _act_sl is not None and (_act_sl.stop - _act_sl.start) > self.action_dim:
            raise ValueError(
                f"obs_spec 的 act 段宽 {_act_sl.stop - _act_sl.start} 超过 "
                f"action_dim={self.action_dim}：底盘给不出这么多动作分量"
            )
        # Box 的上下界由各段自己推导（obs/spec.py 的 Segment.bounds），env 只负责
        # 把环境量喂进去。没有推导依据的段（Custom、未声明量程的 lidar 传感器）
        # 保持 ±inf —— 不猜；那种情况下 check_env 会点名，这是信号不是缺陷。
        _b_low, _b_high = self._obs_resolved.bounds(self._bound_context())
        if priv_dim:
            # privileged 段是 PrivilegedEncoder 的 [0,1] 归一化编码。
            _b_low = np.concatenate([_b_low, np.zeros(priv_dim, dtype=np.float32)])
            _b_high = np.concatenate([_b_high, np.ones(priv_dim, dtype=np.float32)])
        self.observation_space = spaces.Box(
            low=_b_low.astype(np.float32),
            high=_b_high.astype(np.float32),
            shape=(self._full_obs_dim,),
            dtype=np.float32,
        )
        # 越界计数：_clip_obs 累加，reset/step 只在非零时写进 info。
        self._obs_clipped = 0

        # render_mode=None ⇒ 不渲染（不 warning，与 gymnasium 一致）；"human" 走
        # 既有 render 子包画到可复用 Figure，"rgb_array" 返回 uint8 (H, W, 3)。
        # fps 由 dt 推，只能写进实例 metadata（类级给默认值）。
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(
                f"render_mode 只能是 None / 'human' / 'rgb_array'，得到 {render_mode!r}"
            )
        self.render_mode: str | None = render_mode
        self._render_fig = None
        self._render_ax = None
        self.metadata = {
            **type(self).metadata,
            "render_modes": ["human", "rgb_array"],
            "render_fps": max(1, int(round(1.0 / float(self.dt)))),
        }

        self.switch_radius = 0.5
        self._step_count = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._eval_goal_count = (
            0  # eval 成功计数（deprecated: trainer 改用 returns>100 阈值法）
        )
        self._target_idx = 0
        self._prev_target_idx = -1
        # 地图回退标记（P2-20）：reset 里置位；info 每步恒定输出（键集合稳定）
        self._map_fallback = False
        self._last_advance_step = 0
        self._prev_v_sign = 0  # 追踪速度方向变化，防止倒车/前进切换时误判卡住
        self._last_nonzero_v_sign = 0  # 上一个非零 v_sign，用于检测跨 stop 的方向切换
        # dt 已在 __init__ 中赋值（桥与 step 共用），此处不再覆盖
        self._stuck_window = max(30, int(5.0 / self.dt))  # 5 秒无推进→判卡住
        self._progress_threshold = 0.3  # stuck 窗口内需累积至少 0.3m 有效推进
        self._cum_progress = 0.0  # 累积沿路径投影位移 (m)，仅计数正向推进
        # path_dir 的上一帧值：路径点重合（运动方向无法判定）时沿用它，
        # 避免在 cusp/重复点处把方向判成随机值 (2026-10-05)
        self._prev_path_dir = 1.0
        # ── 档位投票滞后 (2026-10-06) ──────────────────────────────────
        # env 内部"这段路该正着开还是倒着开"的**稳定**档位（奖励闸门/偏航权重用）。
        # raw path_dir 在 cusp/自接近区每步抖动 27%~32%（真 stage3/4 实测），
        # 直接当奖励闸门 = 惩罚噪声：正确倒车的车也被随机判违规、推进被清零。
        # 滞后只作用于奖励侧判定与 info，**不进观测**。
        # votes<=1 = 关闭（逐位回到旧行为）。
        # 兜底值同样对齐 EnvConfig.gear_lag_votes 的默认（3）。这是历史遗留：
        # 兜底写 1（= 关闭滞后）与 dataclass 默认 3 不一致，只在传入缺该字段的
        # 旧 EnvConfig 时生效。想关滞后请在 config 里显式写 gear_lag_votes: 1。
        self._gear_lag_cfg = int(getattr(_ec, "gear_lag_votes", 3))
        self._gear_lag = GearVoteLag(self._gear_lag_cfg)
        self._prev_position = np.array([0.0, 0.0], dtype=np.float32)
        self._arc_lengths = np.zeros(1, dtype=np.float64)  # 全路径弧长表 (v5)
        self._prev_arc_pos = 0.0  # 上一步弧位置 (v5)
        self._prev_ni = -1  # 上一步最近路点索引 (v5, 窗口化搜索)
        self._prev_goal_dist = None  # 上一步终点距离 (v5, r_goal_prox 势能, B7 修复)

    @property
    def curriculum_stage(self) -> int:
        """当前课程阶段索引（从0开始），如果没有课程则返回 -1。

        B6 修复：env 侧的显式 stage_index 优先（make_env / update_curriculum 注入），
        仅当未设置（<0）时才回退读取 map_generator.stage_index。
        此前只读 map_generator 且 make_env 未传 stage_index → 恒 -1 →
        reward 侧 current_stage=max(-1,0)=0 → stage_scale 恒 0.2。
        """
        if self._stage_index >= 0:
            return self._stage_index
        if self.map_generator is not None:
            return getattr(self.map_generator, "stage_index", -1)
        return -1

    def reset(self, seed=None, options=None):
        """重置环境；options 的优先级只有一条规则（1.1.0 E）。

        options 显式给出（含空 dict `{}`）⇒ **一律以调用方为准**；options 为 None
        且 self.default_reset_options 非空 ⇒ 自动用它；否则按既有的内建场景
        （map_generator / 直线兜底）走。因此 cs.make(scenario=...) 造出来的 env
        直接 reset() 就生效，而显式 options 仍然覆盖它（不会被默认值掺入）。
        """
        if options is None:
            _defaults = getattr(self, "default_reset_options", None)
            if _defaults:
                options = _defaults
        try:
            return self._reset_impl(seed=seed, options=options)
        except Exception:
            import datetime
            import os
            import traceback

            crash = (
                f"\n{'=' * 60}\n"
                f"Worker PID={os.getpid()} reset crash "
                f"{datetime.datetime.now()}\n"
                f"{traceback.format_exc()}"
            )
            try:
                with open(_CRASH_LOG, "a", encoding="utf-8") as f:
                    f.write(crash)
            except Exception:  # noqa: BLE001, S110 - 写崩溃日志本身失败也不能吞掉原始异常
                pass
            raise

    def _reset_impl(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # ── 处理来自 trainer 的课程更新（通过 setattr 推送）──
        pending = getattr(self, "_pending_curriculum", None)
        if pending is not None:
            self.update_curriculum(pending)
            self._pending_curriculum = None

        self._step_count = 0
        self._ctx_builder.reset()  # 清除时序 path 上下文
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._last_delta = 0.0
        self._target_idx = 0
        self._prev_target_idx = -1
        self._last_advance_step = 0
        self._cum_progress = 0.0  # 累积正向推进距离
        self._prev_path_dir = 1.0  # path_dir 历史值跨 episode 重置
        self._gear_lag.reset()  # 档位投票滞后状态跨 episode 重置 (2026-10-06)
        self._prev_v_sign = 0  # 重置速度方向追踪
        self._last_nonzero_v_sign = 0  # 重置上一个非零方向，防止跨 episode 状态泄漏
        if self._safety_gate is not None:
            self._safety_gate.reset()  # 清除 safety gate 的 prev_delta/prev_v

        # 地图生成重试上限（P2-20）：每个 stage 的自然失败率很低（实测 stage5
        # 约 1/40 次 reset 需要重试），100 次纯属浪费——失败时每轮都要重新跑
        # Hybrid-A*/RS 规划（单次几十~几百 ms），16 个 worker 叠起来就是秒级卡顿。
        # 取 8：单次失败率 ~1/40 时，连续 8 次全失败的概率 ≈ (1/40)^8 ≈ 0
        # ⇒ 能落到绝对回退的基本都是**真无解**的配置（某个种子下迷宫确实排不出路径），
        # 而不是"运气差"。再多重试只是烧时间，不会改结果。
        # ── 场景来源优先级（W4b）：options 覆盖 > map_generator > 内建直线 ──
        self._map_fallback = False
        _opts = dict(options or {})
        _override = any(
            _opts.get(_k) is not None
            for _k in ("obstacles", "path_waypoints", "initial_heading")
        )
        if _override:
            obstacles = _opts.get("obstacles")
            path = _opts.get("path_waypoints")
            initial_heading = _opts.get("initial_heading")
        elif self.map_generator is not None:
            # goal_heading 由桥按路径末端自行推导，这里的返回值用不上。
            obstacles, path, _goal_heading, initial_heading = self._generate_scene()
        elif self._default_scenario_active(options):
            # 3.0.0：不给 options 时接 env 自带场景（不再退化成固定直线）。
            # 它返回的是完整 reset options，于是下面的动态障碍分支照常生效。
            _opts = self._default_scene(seed)
            obstacles = _opts.get("obstacles")
            path = _opts.get("path_waypoints")
            initial_heading = _opts.get("initial_heading")
        else:
            # **有意差异**：原实现这里传 (None, None, 0.0, None)，让桥用
            # 20 点 10m 的隐式直线兜底；这里显式给出同一条直线（1,5 → 9,5，
            # 0.05 m 间隔），使场景在 reset 后可观测，几何完全一致。
            obstacles, path, _goal_heading, initial_heading = _straight_scene()

        self._obstacles = obstacles or []
        # 动态障碍：**仅当** payload 真的带这个键时才下发（键不存在则一行行为都
        # 不变，静态场景 reset 路径与加入动态障碍前完全等价）。位置/相位由桥在
        # reset 里归零，因此下一回合从 t=0 的初态开始。
        if "dynamic_obstacles" in _opts:
            _dyn = list(_opts.get("dynamic_obstacles") or [])
            self.bridge.set_dynamic_obstacles(_dyn)
        elif self._dynamic_obstacles:
            # 从动态场景切回静态场景：显式清空，否则上一回合的动态障碍会残留。
            _dyn = []
            self.bridge.set_dynamic_obstacles(_dyn)
        else:
            # 从来没用过动态障碍 ⇒ 一个调用都不发（静态路径逐位等价）
            _dyn = None
        if _dyn is not None:
            self._dynamic_obstacles = [dict(_spec) for _spec in _dyn]
        # 跨回合清空 last-valid scan 缓存：否则上一回合末帧会被零阶保持进新回合。
        # 必须在 bridge.reset() 之前清 —— reset 会产出首帧，_build_obs 随后把它写回。
        self._lidar_last_scan.clear()
        self.bridge.reset(
            obstacles=obstacles, path_waypoints=path, initial_heading=initial_heading
        )
        if self._dynamics is not None:
            # 每回合从静止起步：不清会把上一回合残留的 v/ω 带进新回合（既不符合
            # "从静止开始"的物理，也破坏"同 seed 两次连跑逐位相同"）。
            self._dynamics.reset()

        raw = self._get_raw_obs()
        self._prev_position = np.array(
            [raw["odometry"][0], raw["odometry"][1]], dtype=np.float32
        )

        # ── 势能型 progress: 全路径弧长表 + 初始弧位置 (2026-08-16 v5) ──
        _pw = self.bridge.get_path_global()
        if len(_pw) >= 2:
            _seg = np.hypot(np.diff(_pw[:, 0]), np.diff(_pw[:, 1]))
            self._arc_lengths = np.concatenate([[0.0], np.cumsum(_seg)])
        else:
            self._arc_lengths = np.zeros(1, dtype=np.float64)
        self._prev_ni = -1  # 窗口化搜索: 首步全路径 argmin 锚定
        self._prev_arc_pos = 0.0  # 无路径 fallback 基线（有路径时被投影值覆盖）
        self._prev_goal_dist = (
            None  # r_goal_prox 势能基线 (B7)：首步 ΔΦ=0，防开局假奖励尖峰
        )
        # 奖励侧回合边界（2026-10-29）：显式清空稠密累计，不再只依赖
        # "steps_taken 严格回退"这一隐式契约（见 RewardCalculator.reset）。
        self.reward_calc.reset()
        _init_pos = np.array([raw["odometry"][0], raw["odometry"][1]])
        self._prev_arc_pos = self._arc_position(_init_pos, _pw)

        prev_delta = self._prev_action[0]
        prev_v = self._prev_action[1]
        # reset 时 _target_idx 刚归零 ⇒ 真值 arc_completion = 0/(n_pts-1) = 0，
        # 显式传 0.0（观测里的进度维在回合起点必须归零）。
        obs_np = self._build_obs(
            raw, prev_delta=prev_delta, prev_v=prev_v, path_progress=0.0
        )
        if self._priv_enabled and self._priv_encoder is not None:
            obs_np = self._append_privileged(obs_np, raw["odometry"])
        obs_np = self._clip_obs(obs_np)
        # 传感器读数与 step 共用 _sensor_info()（唯一口径）：桥在 reset 末尾已产出
        # 真实首帧（S3b，见 SimpleBridge.reset），所以这里拿到的是本回合 t=0 的读数，
        # 不是空字典；未启用传感器时仍是 {}（不新增任何键）。
        return obs_np, self._reset_info()

    # ── 车型适配层（W4b）──────────────────────────────────────────────
    def _v_norm(self, v: float) -> float:
        """速度 → 动作标度；与 command() 的 a1>=0 → *v_max else *(-v_min) 互逆。"""
        if v >= 0.0:
            return float(v / max(self.v_max, V_EPS))
        return float(v / max(-self.v_min, V_EPS))

    def _steer_from(self, v: float, omega: float) -> float:
        """转向维标度：阿克曼 = 物理转角 δ；差速/麦轮 = ω/ω_max·delta_max。

        零速时 δ 无法由 (v, ω) 反解（运动学上 ω 必须同时为 0），沿用上一拍
        转角 —— 与 controllers/base.py:to_chassis_action 同口径。只在
        accel/steer_rate/safety_gate 这类非默认路径上才会走到。
        """
        if self.chassis_name == "ackermann":
            if abs(v) >= V_EPS:
                return float(np.arctan(omega * self.wheelbase / v))
            return float(self._last_delta)
        return float(omega / max(self.omega_max, V_EPS) * self.delta_max)

    def _omega_from_steer(self, steer: float, v: float) -> float:
        """上式逆映射：阿克曼 δ → ω（零速保持 0，与 ackermann.py:step 一致）。"""
        if self.chassis_name == "ackermann":
            return 0.0 if v == 0.0 else float(v / self.wheelbase * np.tan(steer))
        return float(steer / max(self.delta_max, V_EPS) * self.omega_max)

    def _encode_physical(self, v: float, omega: float, vy: float = 0.0) -> np.ndarray:
        """物理量 → 归一化动作（与 controllers/base.py:to_chassis_action 同公式）。

        这段逆映射与 controllers.base.to_chassis_action 是**同一条公式的两份
        实现**：后者是 Controller 的方法而 Controller 是 ABC（无法在 env 里
        实例化复用），且 env 在无控制器时也要能工作。改一处务必同步另一处。
        """
        act = np.zeros(self.action_dim, dtype=np.float32)
        if self.chassis_name == "ackermann":
            delta = self._steer_from(v, omega)
            self._last_delta = delta
            if self.curvature_action:
                act[0] = np.tan(delta) / max(self.kappa_max * self.wheelbase, V_EPS)
            else:
                act[0] = delta / max(self.delta_max, V_EPS)
            act[1] = self._v_norm(v)
        elif self.action_dim >= 3:  # mecanum: [v, vy, ω]
            act[0] = self._v_norm(v)
            act[1] = vy / max(self.vy_max, V_EPS)
            act[2] = omega / max(self.omega_max, V_EPS)
        else:
            # 动作空间已统一为归一化 ±1（差速/麦轮同契约，W15）：command() 里
            # v = a0·v_max（a0<0 时 ·(−v_min)）、ω = a1·ω_max ⇒ 这里做
            # 物理量 → 归一化的逆映射（与 controllers/base.py:to_chassis_action
            # 同公式）。恒等透传只在 v_max == 1.0 / omega_max == 1.0 时偶然成立。
            act[0] = self._v_norm(v)
            act[1] = omega / max(self.omega_max, V_EPS)
        return np.clip(act, self.action_space.low, self.action_space.high)

    def _physical_action(self, act: np.ndarray) -> np.ndarray:
        """本步物理动作向量 [转向标度, v, (vy)]：_prev_action / 奖励 / info 的口径。

        ⚠ 这是 **env 内部口径**，不是对外物理向量：差速/麦轮的转向维是
        `_steer_from()` 给出的"转向标度"（ω 归一化到 ±delta_max），obs 的 act 段、
        `_prev_action[0]` 的限幅（:922/:1282）都按它；对外的物理向量是
        `Chassis.physical_action()`（差速 `[v, omega]`、麦轮 `[v, vy, omega]`），
        两者在差速/麦轮上不同，不可互换。阿克曼两者一致（转向维都是物理 δ）。
        """
        cmd = self.chassis.command(act)
        out = np.empty(self.action_dim, dtype=act.dtype)
        if self.chassis_name == "ackermann":
            # 走底盘的公开契约，不再读私有 self.chassis._prev_delta：外部桥只调
            # command()、从不推进底盘状态，_prev_delta 会恒为 0（mapped_action[0]
            # 恒 0 的真缺陷）。physical_action() 直接取本帧 _map_action 的 δ，
            # v=0 时 δ 仍是指令值，且与 SimpleBridge 路径逐位同值。
            out[0] = float(self.chassis.physical_action(act)[0])
        else:
            out[0] = self._steer_from(cmd.v, cmd.omega)
        out[1] = cmd.v
        if self.action_dim >= 3:
            out[2] = cmd.vy
        return out

    def _apply_limits_and_gate(
        self, act: np.ndarray, cmd
    ) -> tuple[np.ndarray, bool, float, float]:
        """加速度/转角速率限幅 + 可选 SafetyGate ⇒ 重新编码成动作。

        非默认路径（三个开关默认全关，此时 step 完全等价于直通）。
        限幅在物理域做（与原实现一致），因此出口要再编码一次。

        返回 (act, safety_brake, v, omega)：后两项是**限幅/gate 之后**的物理量，
        供 dynamics 档（wheel）在同一物理域里接一阶滞后——调用方不必再从 act
        反解（反解要经过 self._last_delta，会把状态读写搅在一起）。
        """
        v = float(cmd.v)
        omega = float(cmd.omega)
        vy = float(cmd.vy)
        prev_steer = float(self._prev_action[0])
        prev_v = float(self._prev_action[1])
        if self._accel_max > 0.0:
            _dv = float(
                np.clip(
                    v - prev_v, -self._accel_max * self.dt, self._accel_max * self.dt
                )
            )
            v = float(np.clip(prev_v + _dv, self.v_min, self.v_max))
        if self._steer_rate_max > 0.0:
            _steer = self._steer_from(v, omega)
            _dsteer = float(
                np.clip(
                    _steer - prev_steer,
                    -self._steer_rate_max * self.dt,
                    self._steer_rate_max * self.dt,
                )
            )
            omega = self._omega_from_steer(prev_steer + _dsteer, v)
        safety_brake = False
        if self._safety_gate is not None:
            delta = self._steer_from(v, omega)
            _frames = self._lidar_frames()
            lidar_360 = (
                np.concatenate([_frames[k] for k in sorted(_frames)])
                if _frames
                else None
            )
            gated = self._safety_gate(np.array([delta, v], dtype=np.float64), lidar_360)
            v = float(gated[1])
            omega = self._omega_from_steer(float(gated[0]), v)
            safety_brake = bool(self._safety_gate.brake_triggered)
        return self._encode_physical(v, omega, vy), safety_brake, v, omega

    def obs_spec_describe(self) -> dict[str, slice]:
        """obs 各段的名字 → slice（与 spec.resolved().describe() 同一份）。

        注意 privileged 尾部（Critic 专用）不在其中：它恒追加在整条向量最末，
        起点 = observation_space.shape[0] - privileged bins。
        """
        return dict(self._obs_slices)

    def describe_dynamics(self) -> dict:
        """动力学档位与**实际生效**的参数（选 limits/wheel 且 YAML 写 0 时看得到默认值）。

        构造器 dynamics= 给过值时多一个 level_source="constructor" 键（说明档位不是
        cfg 选的）；dynamics=None 时字典与加入该参数前逐键逐值相同。
        """
        return dict(self._dynamics_info)

    def _build_obs(
        self,
        raw: dict,
        *,
        prev_delta: float = 0.0,
        prev_v: float = 0.0,
        err_signals: np.ndarray | None = None,
        goal_signals: np.ndarray | None = None,
        path_progress: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """raw → ObsContext → obs_spec → 扁平 float32（不含 privileged 尾部）。"""
        ctx = self._ctx_builder.build_context(
            raw,
            prev_delta=prev_delta,
            prev_v=prev_v,
            err_signals=err_signals,
            goal_signals=goal_signals,
            path_progress=path_progress,
            lidar=self._lidar_frames(),
        )
        ctx.act = self._act_vector(ctx.act)
        return self._obs_resolved.build(ctx)

    def _act_vector(self, base: np.ndarray) -> np.ndarray:
        """按 obs_spec 的 act 段宽裁剪/扩展归一化动作。

        ContextBuilder 只算得出 2 维（[转向标度, v]），3 维车型的第 3 维（横移标度
        _prev_action[2]/vy_max）在这里补齐 —— 补在 act 段**末尾**，与 v0.1.4
        的 _with_chassis_acts（0.2.0 已删）是同一个位置、同一个值（495 维因此逐位不变）。
        act 段宽 ≤ 2 时按宽度截断（spec 明确只要前几维）。
        """
        act_sl = self._obs_slices.get("act")
        if act_sl is None:
            return base
        width = act_sl.stop - act_sl.start
        if width <= base.size:
            return base[:width]
        vy_norm = 0.0
        if self._prev_action.shape[0] >= 3:
            vy_norm = float(self._prev_action[2] / max(self.vy_max, V_EPS))
        fill = np.full(width - base.size, vy_norm, dtype=base.dtype)
        return np.concatenate([base, fill])

    def _lidar_frames(self) -> dict[str, np.ndarray]:
        """传感器套件里 Lidar 的当帧读数（名字 → ranges），供 lidar 段引用。

        只收带 "scan" 键的读数（Imu / WheelOdom 的读数没有它）。缺帧（丢帧或
        latency 预热期，scan 为 None）时的契约（S3b）：

        1. 该传感器**有过**有效帧 ⇒ 交付缓存里最后一帧（零阶保持）。观测的雷达段
           因此与上一帧逐字节相同，绝不因为一次丢帧就谎报"四周空旷"；
        2. **从未**有过有效帧 ⇒ 用 scan_meta["range_max"] 填满。雷达语义里"这一帧
           没有任何回波"就等于打到量程外，塞 NaN 会污染整条观测。这只是兜底：
           正常回合里 bridge.reset() 的首帧已经写满缓存（见 SimpleBridge.reset）；
        3. 缓存只存**真实**读数（兜底填充值不进去，否则一次预热期就会把假数据
           永久保持下去），且只在 env.reset() 里清空，跨回合不残留上一回合的扫描。
        """
        extra: dict[str, np.ndarray] = {}
        getter = getattr(self.bridge, "get_sensor_reading", None)
        reading = getter() if callable(getter) else None
        for name, item in (reading or {}).items():
            if not isinstance(item, Mapping) or "scan" not in item:
                continue
            key = str(name)
            scan = item["scan"]
            if scan is not None:
                held = np.array(scan, dtype=np.float32, copy=True)
                extra[key] = held
                self._lidar_last_scan[key] = held
                continue
            held = self._lidar_last_scan.get(key)
            if held is not None:
                extra[key] = held.copy()
                continue
            meta = item.get("scan_meta") or {}
            extra[key] = np.full(
                int(meta.get("n_rays", 0)),
                float(meta.get("range_max", 0.0)),
                dtype=np.float32,
            )
        # 桥没给读数（例如注入的桥没有 get_sensor_reading）：有历史就保持，
        # 否则按"无回波 = 打到量程外"填满；宽度由 spec 自己声明，绝不靠猜。
        for spec in self._lidar_specs():
            key = str(spec.name)
            if key in extra:
                continue
            held = self._lidar_last_scan.get(key)
            if held is not None:
                extra[key] = held.copy()
            else:
                extra[key] = np.full(
                    int(spec.n_rays), float(spec.max_range), dtype=np.float32
                )
        return extra

    def _sensor_info(self) -> dict[str, Any]:
        """本回合的传感器读数（传感器名 → 读数），step / reset 的唯一口径。

        传感器只经 info 曝光，obs 维度不变（见 bridges/sensors.py 的模块说明）。
        getattr 兜底：env 允许注入任意 duck-typed 桥，未实现该非抽象 API 的桥
        视为"未启用传感器" ⇒ 返回空 dict，调用方因此不新增任何键。
        """
        getter = getattr(self.bridge, "get_sensor_reading", None)
        if getter is None:
            return {}
        return dict(getter() or {})

    def _sensor_meta(self) -> dict[str, dict[str, int]]:
        """lidar 段补射线数用的元数据：每支已装配 Lidar 的名字 → n_rays。

        1.0.0 起没有任何隐式"前/后两键"：
        雷达帧只来自 bridge.get_sensor_reading()，所以这里的键集就是"真实装配的
        雷达"。注入 bridge= 时 env 看不到桥里套件的声明，那种用法下 spec 引用
        套件雷达就必须自给 ray_counts（否则 resolved() 会点名报错）。
        """
        meta: dict[str, dict[str, int]] = {}
        for spec in self._lidar_specs():
            meta[str(spec.name)] = {"n_rays": int(spec.n_rays)}
        return meta

    def _lidar_specs(self) -> tuple[Lidar, ...]:
        """env 能看到的全部 Lidar spec：构造器给的 + 桥内套件自带的（若有）。

        注入 bridge= 时 env 拿不到套件声明，只能鸭子类型读桥里的
        _sensors.specs（SensorSuite 的属性）；读不到就是空集，那种用法下
        引用套件雷达必须自己写 ray_counts。
        """
        candidates: list[Any] = list(self._sensor_specs)
        suite = getattr(self.bridge, "_sensors", None)
        candidates.extend(getattr(suite, "specs", None) or ())
        by_name: dict[str, Lidar] = {}
        for spec in candidates:
            if isinstance(spec, Lidar):
                by_name[str(spec.name)] = spec
        return tuple(by_name.values())

    def _generate_scene(self):
        """跑 map_generator，只吞 MapGenerationError（按类名判定，不 import planning）。

        连续 MAP_GEN_RETRIES 次失败 ⇒ _map_fallback=True + RuntimeWarning +
        回退内建直线场景（与原实现的对应分支行为一致）。
        """
        last_exc = None
        for _ in range(self.MAP_GEN_RETRIES):
            try:
                return self.map_generator.generate(self.np_random)
            except Exception as exc:
                if not _is_map_generation_error(exc):
                    raise
                last_exc = exc
        self._map_fallback = True
        warnings.warn(
            f"map_generator 连续 {self.MAP_GEN_RETRIES} 次抛 MapGenerationError"
            f"（最后一次：{last_exc}），回退内建直线场景",
            RuntimeWarning,
        )
        return _straight_scene()

    def _project_nearest(
        self, pos: np.ndarray, path_world: np.ndarray
    ) -> tuple[int, float, float] | None:
        """窗口化最近点投影 —— 弧位置与偏离判定共用的唯一口径（P2-17）。

        返回 (ni, t, seg_len)：最近路点索引、段内投影比例 [0,1]、该段长度(m)。
        无可用路径（无弧长表 / 点不足 2 个）时返回 None，调用方自己决定回退语义。

        窗口化搜索（_prev_ni ±25 点 ≈ ±1.25m，机器人每步 ≤0.12m 绰绰有余）：
        路径自我接近（stage4 折返段 / stage5 迷宫平行走廊）时全路径 argmin 最近点
        可跨段跳变（单步 arc 跳 ±35m → ±140 奖励尖峰），窗口化保证沿路径连续性。
        """
        n = len(path_world)
        if not (len(self._arc_lengths) > 1 and n >= 2):
            return None
        if len(self._arc_lengths) != n:
            # 长度守卫: 路径与弧长表不一致时重建（防 IndexError）
            _seg = np.hypot(np.diff(path_world[:, 0]), np.diff(path_world[:, 1]))
            self._arc_lengths = np.concatenate([[0.0], np.cumsum(_seg)])
            self._prev_ni = -1  # 路径已变，重锚定
        if self._prev_ni >= 0:
            lo = max(0, self._prev_ni - 25)
            hi = min(n - 1, self._prev_ni + 25)
            _d = np.hypot(
                path_world[lo : hi + 1, 0] - pos[0], path_world[lo : hi + 1, 1] - pos[1]
            )
            _ni = lo + int(np.argmin(_d))
        else:
            _d = np.hypot(path_world[:, 0] - pos[0], path_world[:, 1] - pos[1])
            _ni = int(np.argmin(_d))
        _ni = min(_ni, n - 2)
        self._prev_ni = _ni
        _seg_vec = path_world[_ni + 1, :2] - path_world[_ni, :2]
        _seg_len = float(np.hypot(*_seg_vec))
        if _seg_len > 1e-9:
            _t = float(
                np.clip(
                    np.dot(pos - path_world[_ni, :2], _seg_vec) / _seg_len**2, 0.0, 1.0
                )
            )
        else:
            _t = 0.0
        return _ni, _t, _seg_len

    def _arc_position(self, pos: np.ndarray, path_world: np.ndarray) -> float:
        """势能型弧位置: 最近路点弧长 + 段内投影 (2026-08-16 v5)。

        投影全部委托给 _project_nearest（与偏离判定同一个窗口、同一个最近点），
        行为与重构前逐位一致。无有效路径时返回 _prev_arc_pos（弧位置不变）。
        """
        proj = self._project_nearest(pos, path_world)
        if proj is None:
            return self._prev_arc_pos
        _ni, _t, _seg_len = proj
        return float(self._arc_lengths[_ni] + _t * _seg_len)

    def _windowed_cte(self, pos: np.ndarray, path_world: np.ndarray) -> float | None:
        """推进窗口内的横向误差 (P2-17) —— 与 _arc_position 同一投影口径。

        返回值符号与 body 系 cte 一致：
            cte = cross(u, w) = cos(th_p)·w_y − sin(th_p)·w_x
        其中 u 是最近路点的**鼻向**单位向量、w 是"路点 − 车"的位移向量。
        该式与 body 系的 -nx·sin(t) + ny·cos(t) 逐位等价（叉积在旋转下不变），
        所以阈值语义（对接 config 的 path_deviation_limit）完全保留。

        为什么不能用全局 argmin 的 cte：
            折返段 / 迷宫平行走廊上，**另一条支路**往往就在车脚下（支路间距 ~1m）。
            车其实已经跑到那条支路上时，全局最近点就是脚下这条支路 ⇒ |cte| ≈ 0
            ⇒ 4m 偏离上限永不触发，episode 永远不会因偏离而终止（缺陷本体：
            偏离判定被另一条支路短路）。窗口化投影锚在**自己这条支路**的历史推进
            位置上，量到的是"离该走的路径有多远"—— 这才是 path_deviation 的本意。
            跨支路在单步内不可能发生：单步位移 ≤ max|v|·dt = 0.12m，窗口 ≥±1.25m
            （与 _advance_target_idx / arc_completion 的等价条件同一套）。

        返回 None 表示当前没有可信投影（无路径 / 首步未锚定），调用方回退到
        body 系全局 argmin 的 cte（= 旧行为）。
        """
        n = len(path_world)
        if not (n >= 2 and 0 <= int(self._prev_ni) < n - 1):
            return None
        _ni = int(self._prev_ni)
        w = path_world[_ni, :2] - pos
        th = float(path_world[_ni, 2]) if path_world.shape[1] >= 3 else 0.0
        return float(np.cos(th) * w[1] - np.sin(th) * w[0])

    def _advance_target_idx(
        self, path_body: np.ndarray, n_pts: int, dists: np.ndarray | None = None
    ) -> int:
        """推进 `_target_idx`：主窗口 + **有界追赶**（2026-10-06 修复）。

        【病象】原实现只在 ``[target_idx, target_idx+15]``（0.75m）内找
        ``d < switch_radius(0.5m)`` 的点，找不到就原地不动。这个条件一旦失败
        就**自我维持**：索引落在车后方越远，窗口离车越远，越不可能命中。
        实测（run001 / ckpt 1.59M / numpy 后端 / 确定性策略 / 120 集）：
        冻结段（索引连续 ≥20 步不变）里车到自己真实投影的横向距离中位只有
        **0.34m**（车其实贴得很好），而窗口内最近点中位 **1.72m** ——
        即冻结的真因是**滞后**（实测滞后 p90 = 90~103 点 ≈ 5m，max 233 点），
        不是横向偏移。后果：obs 第 72 维 `path_progress` 长时段是错的常数、
        timeout 惩罚 `-100×(1-arc_completion)` 同步失真。

        【修法】把"推进"拆成两件事，各自的界都是**物理量**：

        1. **可达性（在哪）**：主窗口 ``[idx, idx+15)`` 内取 ``d < switch_radius``
           的点里**离车最近**的那个作投影；窗口内没有命中时，整条前向路径上找
           "同向 + 局部"的候选点（``|hel| ≤ CATCHUP_HEADING_TOL``、
           ``d ≤ R_eff``），同样取**离车最近**的作投影 ``i_proj``。
           **两个分支都只取"车已经到过的位置"（真实投影）：不做前瞻、不强制 +1**，
           候选集也**不设前向掩码**（旧实现的 ``cand[:idx+1] = False`` 等于"下一个点"），
           单调性完全交给 ``max(idx, ...)``。
        2. **限速（走多快）**：``_target_idx`` 单步最多沿路径推进
           ``max|v|·dt·K``（K=5 ⇒ 0.5m，物理步长的个位数倍），
           且**只增不减**（``max(idx, ...)``）。

        【P1-2 二次修复 2026-10-06：进度维恒超前】原实现的"前瞻"让
        ``arc_completion`` 指向车**还没到**的地方：主窗口取最远命中点再 ``+1``、
        追赶分支强制在投影点索引上额外 ``+1``。实测（真 stage3/4/5 地图 + numpy 后端）：
        rollout 里 ``arc_completion`` 比真实弧长进度 **100% 偏高**
        （p50 +0.065、max +0.149）；机器人**静止 40 步**在 stage4 上就能爬到
        0.03~0.18。这一维同时是观测最后一维与 timeout 惩罚
        ``-100×(1-arc_completion)`` 的输入，而奖励侧的 ``arc_progress`` 走
        ``_arc_position``（正确的投影口径）⇒ 同一回合两套进度互相矛盾。
        现在两个分支都取**最近**命中点、都不做 ``+1``，口径与 ``_arc_position``
        统一为"路径上离车最近的点"（等价条件见 ``_step_impl`` 里
        ``arc_completion`` 处的注释）。

        【已知残余（未修：方向判据失效的场景）】主窗口的判据是**纯距离**
        （既定语义：窗口内 ``d < switch_radius``），所以当路径在 0.5m 内
        **同鼻向**地折回来时，车被推到折回支路附近就会沿该支路慢慢推进。
        实测真 stage4 一幅图（专家开 100 步、车始终没离开路径起点）：索引仍涨到
        15~20 点 = arc_completion 0.067~0.089（修复前同一条轨迹 0.116）。试过给主窗口也加
        ``same_dir`` 判据：**不解决**（该折回段鼻向相同），反而让正常推进滞后
        （实测 s5/专家 100 步 0.207 → 0.047），故未采用。

        【为什么不会跨支路跳变】

        * **单步位移有界**：任何一次推进的 Δarc ≤ `max|v|·dt·K` = 0.5m。
          折返/迷宫平行支路的间距 ~1m，跨支路至少要跨过 ≥1m 弧长 ⇒
          **单步内不可能完成**（实测对照："全局 argmin 兜底"的错法单步
          Δarc 达 11.6m = k=116）。
        * **投影点必须"局部"**：``d ≤ R_eff``。半径是**自适应**的
          ``R_eff = clip(max(switch_radius, d_own+0.25), ·, 1.25)``，其中
          ``d_own`` = 车到**同向**路点的最小距离。注意 ``d_own`` 只用**距离标量**、
          不用其索引 ⇒ 即便最近的同向点在另一条平行支路上，量出的半径只会
          **偏小**（更保守），误差方向是安全侧。
        * **投影点必须"同向"**：``|hel| ≤ 1.2 rad``（`hel` = 路径鼻向 − 车头朝向，
          即 body 系路径第 3 列）。折返/回头的另一条支路与车头朝向相差 ~π，
          被这条判据挡掉；而同向的平行支路要同时满足"横向 ≤1.25m"才可能被误选，
          再叠加单步 0.5m 限速 ⇒ 最坏是一段**缓变**（每步 0.5m、单调）而不是跳变。
        * 主窗口（≤15 点、0.5m）**保持原判据**（半径 switch_radius、15 点）：
          窗口越窄越保守，越宽才需要越严；它的推进同样被预算钳到 ≤0.5m/步。
          判据不变，只是由"最远命中点 +1"改成"最近命中点"（P1-2）。

        与 B2 的单调约定一致：本函数返回值恒 ≥ 传入的 `_target_idx`。
        """
        idx = int(self._target_idx)
        if n_pts <= 0:
            return idx
        idx = min(max(idx, 0), n_pts - 1)
        if n_pts < 2 or idx >= n_pts - 1:
            return idx

        # ── 弧长表（body 系与 world 系刚体变换保距 ⇒ 可直接由 path_body 求）──
        _seg = np.hypot(np.diff(path_body[:, 0]), np.diff(path_body[:, 1]))
        _arc = np.concatenate([[0.0], np.cumsum(_seg)])
        # ── 单步弧长预算（含"至少 2 个路点"下限，防粗采样路径上永不前进）──
        _v_scale = max(abs(float(self.v_max)), abs(float(self.v_min)))
        budget = max(
            _v_scale * float(self.dt) * self.CATCHUP_ARC_K, 2.0 * float(_seg[idx])
        )
        i_cap = int(np.searchsorted(_arc, _arc[idx] + budget, side="right")) - 1
        i_cap = int(np.clip(i_cap, idx + 1, n_pts - 1))

        if dists is None:
            dists = np.hypot(path_body[:, 0], path_body[:, 1])

        # ── ① 主窗口：判据不变（半径 switch_radius、15 点），取**最近**命中点 ──
        # 旧实现取窗口内**最远**的命中点再 +1（前瞻）⇒ _target_idx 指向车还没到
        # 的地方，arc_completion 系统性超前（见 docstring 的 P1-2）。现在只在命中
        # 点里取 dists 最小者 = 车到参考路径的**真实投影**路点。
        # 【已知残余】窗口不带方向判据（既定语义）：若路径在 0.5m 内**同鼻向**地
        # 折回来，车被推到折回支路附近仍会沿该支路慢慢推进（实测真 stage4 一幅图、
        # 专家 100 步内车始终没离开路径起点，arc_completion 最终 0.067~0.089 —— 修复前
        # 同一条轨迹 0.116）。试过给窗口也加 same_dir 判据：**不解决**（该折回段
        # 鼻向相同），反而让正常推进滞后（实测 s5/专家 100 步 0.207 → 0.047），
        # 故不加，留作已知残余（详见函数 docstring）。
        look_end = min(idx + 15, n_pts)
        _win = dists[idx:look_end]
        _hit = np.where(_win < self.switch_radius)[0]
        if len(_hit) > 0:
            i_proj = idx + int(_hit[int(np.argmin(_win[_hit]))])
            return int(min(max(idx, i_proj), i_cap))

        # ── ② 有界追赶：整条前向路径上找"同向 + 局部"的最近点作投影 ──
        if path_body.shape[1] >= 3:
            _hel = np.abs(path_body[:, 2])
        else:  # 无鼻向信息 ⇒ 不做方向判据
            _hel = np.zeros(n_pts, dtype=path_body.dtype)
        same_dir = _hel <= self.CATCHUP_HEADING_TOL
        if not same_dir.any():
            return idx  # 车头与整条路都不同向：fail-safe
        d_own = float(dists[same_dir].min())  # 车到"自己这条同向支路"的横向距离
        radius = min(
            max(self.switch_radius, d_own + self.CATCHUP_MARGIN),
            self.CATCHUP_RADIUS_MAX,
        )
        cand = same_dir & (dists <= radius)
        # 旧实现这里还叠了一条 ``cand[:idx + 1] = False``（只允许向前）：与下面的
        # ``+1`` 叠加后净效果是"候选集里最靠前的点 +1"，即便车**静止**在自己的
        # 投影上也会每步前进一个点（横向偏出 0.5m 的悬停车会把半径内的点爬完）。
        # 现在不设前向掩码：单调性完全由 max(idx, ...) 提供——投影在车后方时
        # 原样返回 idx（不前进、也不回退）。
        if not cand.any():
            return idx  # 没有可信候选：fail-safe
        i_proj = int(np.argmin(np.where(cand, dists, np.inf)))
        # 限速：本步最多推进到预算上限。**不再强制 +1** —— 旧实现在投影点索引上
        # 额外加 1（即 ``min(max(idx, 投影索引 + 1), i_cap)`` 的写法），会让车
        # **静止**时也每步前进一个点（哪怕投影就是车当前所在路点），是超前量的
        # 第二个来源。单调性仍由 max(idx, ...) 保证：投影 == idx 时原样返回。
        return int(min(max(idx, i_proj), i_cap))

    def step(self, action: np.ndarray):
        try:
            return self._step_impl(action)
        except Exception:
            import datetime
            import os
            import traceback

            crash = (
                f"\n{'=' * 60}\n"
                f"Worker PID={os.getpid()} step crash "
                f"{datetime.datetime.now()}\n"
                f"{traceback.format_exc()}"
            )
            try:
                with open(_CRASH_LOG, "a", encoding="utf-8") as f:
                    f.write(crash)
            except Exception:  # noqa: BLE001, S110 - 同上：日志写入失败不得掩盖原始异常
                pass
            raise

    def _step_impl(self, action: np.ndarray):
        self._step_count += 1

        # 动作通路（W4b）：mapped_action 的物理映射交给底盘，env 只做
        # 保 dtype 的拷贝 + 形状校验 + 可选限幅/闸门。
        # 「保 dtype」很关键：原实现 mapped_action = action.copy() 后逐元素覆盖，
        # dtype 跟输入走；对拍要位级一致就不能在这里强制 float32。
        act = np.array(action, copy=True)
        if act.shape != (self.action_dim,):
            raise ValueError(
                f"action 形状必须是 {(self.action_dim,)}，收到 {act.shape}"
            )
        cmd = self.chassis.command(act)  # 纯映射：动作 → 物理量
        v_ph = float(cmd.v)
        omega_ph = float(cmd.omega)
        safety_brake = False
        if (
            self._accel_max > 0.0
            or self._steer_rate_max > 0.0
            or self._safety_gate is not None
        ):
            act, safety_brake, v_ph, omega_ph = self._apply_limits_and_gate(act, cmd)
        if self._dynamics is not None:
            # wheel 档：同一物理域上的一阶滞后 / 死区 / 轮滑。
            # 顺序：command → 限幅(硬约束) → **本处滞后** → 编码 → bridge.step。
            # 放在限幅之后，是因为滞后是收缩算子 ⇒ |Δv| ≤ accel_max·dt 仍然成立
            # （_prev_action 记的是滞后后的值）；放在前面会把滞后项削掉（见
            # envs/dynamics.py 的模块 docstring）。放在 gate 之后，是因为 gate 是
            # 紧急覆盖，不能让它刹的车被滞后 0.1~0.15 s。
            _v_dyn, _w_dyn = self._dynamics.step(v_ph, omega_ph)
            if self._steer_rate_max > 0.0:
                # 滞后后再反解出的 δ 会越过速率界：v 与 ω 的时间常数不同
                # （v_tau=0.15 > omega_tau=0.10），低速端 arctan(ω·L/v) 把 ω 的滞后差
                # 放大成更大的转角差。实测（默认档、单步满舵）|Δδ| = 0.3821 rad > 界
                # 0.3000 rad，只出现在第 0 步。转向执行器的速率限制在物理上位于模型
                # 最下游 ⇒ 这里对 δ 再夹一次，并用夹后的 δ 反算 ω，保证 wheel 档严格
                # 包含 limits 档的两条界（附测试 test_wheel_level_keeps_steer_rate_bound）。
                _prev_steer = float(self._prev_action[0])
                _steer = self._steer_from(_v_dyn, _w_dyn)
                _dsteer = float(
                    np.clip(
                        _steer - _prev_steer,
                        -self._steer_rate_max * self.dt,
                        self._steer_rate_max * self.dt,
                    )
                )
                _w_dyn = self._omega_from_steer(_prev_steer + _dsteer, _v_dyn)
            act = self._encode_physical(_v_dyn, _w_dyn, float(cmd.vy))

        self.bridge.step(act)

        raw = self._get_raw_obs()
        prev_delta = self._prev_action[0]
        prev_v = self._prev_action[1]
        # 物理动作向量 [转向标度, v, (vy)]：阿克曼转向维由底盘 physical_action() 给出
        mapped_action = self._physical_action(act)
        self._prev_action = mapped_action.copy()

        path_body = raw["path"]
        n_pts = len(path_body)
        # 全路径"车到路点"距离：CTE 与本步索引推进（_advance_target_idx）共用，
        # 只算一次（n_pts 量级，可忽略）。
        _dists = np.hypot(path_body[:, 0], path_body[:, 1]) if n_pts > 0 else None

        # ── CTE 和 heading error（基于最近路径点）──
        # 在 build() 之前计算，作为 err_signals 传入避免观测上下文构建器
        # 内部用 PathExtractor 子采样路径重复计算（两者使用不同路径表示）
        if n_pts > 0:
            dists = _dists
            nearest_idx = int(np.argmin(dists))
            nx, ny = path_body[nearest_idx, 0], path_body[nearest_idx, 1]
            tangent = float(path_body[nearest_idx, 2])
            cte = -nx * np.sin(tangent) + ny * np.cos(tangent)
            heading_error = tangent
            cte_norm = np.clip(cte / self._ctx_builder.path_d_max, -1.0, 1.0)
            err_signals = np.array([cte_norm, heading_error], dtype=np.float32)

            # heading_error_local = 路径鼻向 − 车头朝向。
            # 【重要】因为 path_body[:,2] 存的是**鼻向**（RS 积分），正确跟踪时
            # 前进段与倒车段它**都 ≈ 0**；只有"该倒车却掉头正着开"时才会 ≈ ±π。
            if n_pts > 0 and path_body.shape[1] >= 3:
                heading_error_local = float(path_body[nearest_idx, 2])
            else:
                heading_error_local = heading_error

            # ── path_dir：这段路该"正着开"还是"倒着开" (2026-10-05 修复) ──
            # 必须用**几何**判定：路径点的位置差 = **运动方向**，而 path_body[:,2]
            # = **鼻向**；两者点积 < 0 ⇒ 倒车段。
            # 原实现 `1.0 if abs(heading_error_local) < π/2 else -1.0` 是错的 ——
            # heading_error_local 正是"鼻向 − 车头"，正确跟踪时前进/倒车段都 ≈0，
            # 于是 path_dir **恒为 +1**，后果：
            #   (a) 倒车段 progress 的符号翻转（下方 tangent_global 取反）永不生效，
            #       正确倒车时 progress 恒为负 → _cum_progress 永不累计（实测 0.00）；
            #   (b) 传给 reward 的 path_dir 失去意义（见 reward.py 同步修复）。
            # 用中心差分取运动方向：比单边差抗噪，且天然处理路径两端。
            path_dir = segment_is_reverse(
                path_body, nearest_idx, heading_error_local, self._prev_path_dir
            )
            self._prev_path_dir = path_dir
            # ── 档位投票滞后 (2026-10-06)：奖励侧用**稳定**档位 ──────────
            # raw path_dir 每步抖动 27%~32%（真 stage3/4 倒车段实测）⇒ 直接拿它
            # 当闸门 = 惩罚噪声（正确倒车也被判违规、arc_progress 被清零），
            # 且"惩罚会随机出现"这件事本身在教策略"别倒车"。
            # path_dir_gate 只流进奖励（闸门 + 倒车段偏航权重），不进观测。
            path_dir_gate = self._gear_lag.update(path_dir)
        else:
            cte = 0.0
            heading_error = 0.0
            heading_error_local = 0.0
            path_dir = self._prev_path_dir
            path_dir_gate = self._gear_lag.update(path_dir)
            err_signals = np.array([0.0, 0.0], dtype=np.float32)

        # ── Goal signals（终点方向预判，传入 observation）──────
        # 在 _build_obs() 前计算，让 actor 感知终点方位。
        # goal_heading_error: 车头指向与终点方向的偏差（sin/cos 两维）
        # 【2026-10-28 删第 3 维】1/(goal_dist+0.5) 是终点距离的确定性函数，
        # 零独立信息，已删除（GOAL_SIG 3→2）。
        if n_pts > 0:
            _goal_dist = float(np.hypot(path_body[-1, 0], path_body[-1, 1]))
        else:
            _goal_dist = 10.0  # 无路径时默认远处
        _robot_theta = float(raw["odometry"][2])
        _goal_heading = float(getattr(self.bridge, "goal_heading", 0.0))
        _ghe = _robot_theta - _goal_heading
        _ghe = (_ghe + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        _goal_signals = np.array(
            [
                np.sin(_ghe),
                np.cos(_ghe),
            ],
            dtype=np.float32,
        )

        # 【观测组装位置】obs 在下面 arc_completion 算完之后再组装（2026-10-05）：
        # 观测里的 path_progress 维必须与**奖励侧**用的是同一个数（同一步、
        # 同一 target_idx），否则折返段会出现"观测说进了、奖励说退了"的错拍。
        # 原位置（此处）在 _target_idx 推进之前，装出来的进度会晚一步。
        # 组装所需的 raw / err_signals / goal_signals / prev_delta / prev_v
        # 全部在本行之前已就绪，移动不改变任何输入。

        collision = self.bridge.check_collision()
        goal_reached = self.bridge.check_goal_reached(self.goal_threshold)

        # 速度方向变化时重置 target_idx 到最近路点（非全局起点 0）。
        # path_body 是 bridge.get_path() 返回的全量路径 (start→goal)，
        # 归零会让机器人突然把"地图起点"当追踪目标。
        # 存 last_nonzero_v_sign 避免 1→0→-1 过渡时丢失方向信息。
        v_sign = (
            1 if mapped_action[1] > 0.01 else (-1 if mapped_action[1] < -0.01 else 0)
        )
        if v_sign != 0 and v_sign != self._last_nonzero_v_sign:
            # ── 速度符号翻转时重同步目标索引 (2026-10-05 修复 + 2026-10-06 收口) ──
            # 原来直接 `= nearest_idx`（全局 argmin），它会**回退**：实测 6 次
            # 翻转有 4 次把 _target_idx 从 112→89、9→0，导致
            # arc_completion 非单调（0.404→0.321、0.040→0.000）⇒ timeout 惩罚
            # 的缩放错乱、推进窗口 [target_idx, +15] 倒退。
            # 2026-10-05 改为 max(_target_idx, nearest_idx) 保住单调，但
            # `nearest_idx` 仍是**全局 argmin** ⇒ 在折返/迷宫平行走廊上会落到
            # 另一条支路：实测单步 Δarc 尖峰 **11.6m / 5.95m（k=116 / 60）**。
            # 2026-10-06：重同步**不再单独做**——紧接其后的
            # `_advance_target_idx` 已经含"同向 + 局部 + 限速"的追赶，
            # 在这里再推一次会让同一步出现两次推进（实测 Δarc 破预算到 0.98m）。
            # 单步预算必须"每步只花一次"，故重同步完全交给那一次调用。
            self._last_advance_step = self._step_count
        self._prev_v_sign = v_sign
        if v_sign != 0:
            self._last_nonzero_v_sign = v_sign

        # 推进目标索引——路径为时间序列，始终单向推进（包括倒车段）。
        # Planner 输出的 waypoints 按时间顺序排列：索引 0=起点，N-1=终点。
        # 无论前进还是倒车，车辆总是沿着索引增大方向完成任务进度。
        # 【2026-10-06 有界追赶】见 `_advance_target_idx`：滞后再深也会每步
        # 最多前进 0.5m 弧长地把投影追上去，不再出现"永久冻结"；但**不强制**
        # 推进——车不动 ⇒ 索引不动（P1-2，见该函数 docstring）。
        if n_pts > 0:
            self._target_idx = self._advance_target_idx(path_body, n_pts, dists=_dists)

        # ── 进度计算 + 曲率预瞄 ──
        if n_pts > 0:
            target_idx = min(self._target_idx, n_pts - 1)

            # 位移投影到目标路点的切线方向（计算沿路径的进度）
            current_pos = np.array([raw["odometry"][0], raw["odometry"][1]])
            displacement = current_pos - self._prev_position
            robot_theta = float(raw["odometry"][2])
            if path_body.shape[1] >= 3 and target_idx < n_pts - 1:
                tangent_body_progress = float(path_body[target_idx, 2])
            elif target_idx < n_pts - 1:
                dx = path_body[target_idx + 1, 0] - path_body[target_idx, 0]
                dy = path_body[target_idx + 1, 1] - path_body[target_idx, 1]
                tangent_body_progress = float(np.arctan2(dy, dx))
            else:
                tangent_body_progress = heading_error
            tangent_global = np.array(
                [
                    np.cos(tangent_body_progress + robot_theta),
                    np.sin(tangent_body_progress + robot_theta),
                ]
            )
            # waypoint theta 为鼻向（RS 积分）。
            # 倒车段需翻转为运动方向——否则 displacement·tangent 恒负 → progress<0。
            if path_dir < 0:
                tangent_global = -tangent_global
            progress = float(np.dot(displacement, tangent_global)) / self.dt

            self._prev_position = current_pos

            # ── 势能型进度: 弧位置 + 物理钳制 (2026-08-16 v5) ──
            _arc_pos = self._arc_position(current_pos, raw["path_world"])
            # 【P2-17】偏离判定用的 cte 必须取**推进窗口内**的投影，而不是 body 系
            # 全局 argmin（后者在折返/平行走廊上会被脚下那条支路压成 ≈0）。
            # 紧跟 _arc_position 调用 ⇒ _prev_ni 已是本步的窗口投影索引，口径统一。
            _cte_win = self._windowed_cte(current_pos, raw["path_world"])
            # 钳制: 机器人每步位移 ≤ v_max×dt = 0.12m，0.3 含量化余量，
            # 兜底路径自我接近时的跨段跳变尖峰（折返段/迷宫平行走廊）。
            arc_progress = float(np.clip(_arc_pos - self._prev_arc_pos, -0.3, 0.3))
            self._prev_arc_pos = (
                _arc_pos  # 用未钳制值更新，防误差累积、保持 telescoping
            )

            # ── 档位闸门 (2026-10-05)：正向进度必须"用对档位"才计入 ──────
            # arc_progress 只看弧长推进、不看是往前开还是往后倒，于是
            # "在倒车段掉头正着开"能白拿和正确倒车一样的 +4.0×arc_progress。
            # 实测 run025：倒车段上 82~90% 的步是反向前进（stage4 82.0%、
            # stage5 89.5%），却横向贴得很紧（|CTE| 0.16~0.27m）——
            # 即"用更省事的方式走完同一条几何曲线"。这里把该补贴归零。
            # 只拦**正向**推进：反向推进（走回头路）仍按原样给负分。
            _v_gear = float(raw["odometry"][3])
            if arc_progress > 0.0 and not gear_is_compliant(path_dir_gate, _v_gear):
                arc_progress = 0.0
        else:
            progress = 0.0
            displacement = np.zeros(2)
            arc_progress = 0.0  # 无路径时不更新弧位置 (v5)
            _cte_win = None  # 无路径 ⇒ 偏离判定回退到旧口径（P2-17）

        # d_min = 所有已装配雷达当帧读数的最小值；一支雷达都没有时取桥的量程
        # （"四周无回波"），绝不拿几何估计冒充读数。
        _lidar_now = self._lidar_frames()
        if _lidar_now:
            d_min = float(min(np.min(f) for f in _lidar_now.values()))
        else:
            d_min = float(getattr(self.bridge, "max_range", 0.0) or 0.0)

        v = float(raw["odometry"][3])
        reverse_triggered = v < -0.01

        # ── 倒车 exploit 防御 (2026-10-05 修复) ─────────────────
        # 防 exploit：车头偏离**路径规定的鼻向** >90° 时没收 progress。
        # 【修复】预期朝向**恒为 0**，不随 path_dir 取 0/π：
        #   path_body[:,2] 是鼻向 ⇒ 正确跟踪时前进段与倒车段的
        #   heading_error_local 都 ≈0。原来的 `0 if path_dir>=0 else π` 配上
        #   "path_dir 由 heading_error_local 反推"会恒等退化成
        #   `min(|h|, π−|h|)` —— 方向盲，"该倒车段掉头正着开"零惩罚。
        #   改成恒取 0 后语义清晰且真正方向敏感：
        #     正确倒车/前进 → h≈0 → dev≈0（不罚）
        #     倒车段却掉头正着开 → h≈±π → dev≈π（罚 ~0.94/步）
        # 注意: 仅没收旧 progress（速率型）。arc_progress 为势能型——
        # 弧位置沿路径时间序天然合法（前进/倒车索引都单调），
        # 振荡净零、方向由弧长符号表达，故有意豁免，无需此处没收。
        _heading_dev = abs((heading_error_local + np.pi) % (2 * np.pi) - np.pi)
        if _heading_dev > np.pi / 2 and progress > 0:
            progress = 0.0

        # 每步位移标量（用于反卡住惩罚）
        disp_mag = float(np.hypot(displacement[0], displacement[1]))

        # 距目标距离 — near_goal 仅用于"目标附近禁用避障惩罚"；
        # r_goal_prox 已是势能型 (B7 修复 2026-08-19)，不再需要门控。
        near_goal = n_pts > 0 and _goal_dist < 2.0
        # r_goal_prox 势能差分需要上一步终点距离：首步/无路径时 prev = 当前值 ⇒ ΔΦ=0，
        # 避免 episode 开头出现一次假奖励尖峰（也保证势能求和严格望远镜化）。
        if n_pts > 0:
            goal_dist = float(_goal_dist)
            goal_dist_prev = (
                float(self._prev_goal_dist)
                if self._prev_goal_dist is not None
                else float(_goal_dist)
            )
            self._prev_goal_dist = float(_goal_dist)
        else:
            goal_dist = 0.0
            goal_dist_prev = 0.0

        # arc_completion: 路点推进比例（target_idx 单调不减，见 _advance_target_idx）
        # 【与奖励侧同口径】arc_completion 与势能型 arc_progress 用的
        # ``_arc_position`` 表达的是**同一个投影概念**——"参考路径上离车最近的
        # 那个点"，只是两个坐标：
        #   * arc_completion = 该投影的**路点索引** / (n_pts-1)
        #   * _arc_position  = 该投影的**弧长**（最近路点弧长 + 段内投影）
        # 实现上必须保留两个函数（一个喂观测/timeout 缩放、一个做势能差分），
        # 其等价条件写死在这里，改任一侧都要重新验证：
        #   ① 同一条路径：都用 bridge 全量路径（时间序，0=起点、N-1=终点）；
        #   ② 两侧都只取**最近点**：无前瞻、无强制 +1（_advance_target_idx）；
        #   ③ 路点间距近似均匀 ⇒ 索引比例 ≈ 弧长比例（实测点距中位 0.0498m、
        #      max 0.05m ⇒ 两者差 < 1 个路点 ≈ 1/(n_pts-1)）；
        #   ④ 投影受 switch_radius 门控：横向偏出 0.5m 时索引冻结
        #      ⇒ arc_completion 相对弧长**滞后**，偏差方向是安全侧。
        # 单调性来自 max(idx, ...)：车走回头路时 _arc_position 会回退，而
        # arc_completion 停在历史最大值（timeout 惩罚缩放要求不回退）。
        arc_completion = self._target_idx / max(n_pts - 1, 1) if n_pts > 1 else 0.0

        # path_remaining: 剩余路径比例，对齐 actor geo_features——同用 nearest_idx
        path_remaining = (
            (n_pts - 1 - nearest_idx) / max(n_pts - 1, 1) if n_pts > 1 else 0.0
        )

        # ── 观测组装（必须在 arc_completion 之后）──────────────────────
        # path_progress 维灌入**当步真值** arc_completion：
        #   * 与奖励侧（timeout 惩罚缩放、info）用的是同一个变量，同一步取值；
        #   * 单调不减（target_idx 只增，见上方"索引只应增大"约定），
        #     折返段/迷宫平行走廊不会像"到终点距离"那样回退；
        #   * 无状态、与 batch 形状无关（旧实现把分母存在 actor buffer 里，
        #     只在 batch=1 更新 ⇒ 生产路径 batch=16 下该维恒为 0）。
        obs_np = self._build_obs(
            raw,
            prev_delta=prev_delta,
            prev_v=prev_v,
            err_signals=err_signals,
            goal_signals=_goal_signals,
            path_progress=arc_completion,
        )
        if self._priv_enabled and self._priv_encoder is not None:
            obs_np = self._append_privileged(obs_np, raw["odometry"])
        obs_np = self._clip_obs(obs_np)

        reward, reward_comps = self.reward_calc.compute(
            done=False,
            collision=collision,
            goal_reached=goal_reached,
            cte=cte,
            heading_error=heading_error,
            d_min=d_min,
            progress=progress,
            v=v,
            v_max=self.v_max,
            prev_delta=float(prev_delta),
            prev_v=float(prev_v),
            delta=float(mapped_action[0]),
            reverse_triggered=reverse_triggered,
            displacement=disp_mag,
            near_goal=near_goal,
            goal_dist=goal_dist,
            goal_dist_prev=goal_dist_prev,
            steps_taken=self._step_count,
            max_steps=self.max_steps,
            arc_completion=arc_completion,
            heading_error_local=heading_error_local,
            path_remaining=path_remaining,
            path_dir=path_dir_gate,  # 奖励侧用滞后稳定档位 (2026-10-06)
            accel_command=float(action[1]),
            current_stage=max(self.curriculum_stage, 0),
            arc_progress=arc_progress,
        )

        # ── 卡住检测 ──────────────────────────────────────────
        # 累积正向推进距离；窗口内累计不足阈值 → 卡住
        # 绕圈/来回振荡虽有绝对速度，但 net progress≈0，逃不过累积检测
        if n_pts > 0 and progress > 0:
            self._cum_progress += progress * self.dt  # 仅计数正向推进
        if self._cum_progress >= self._progress_threshold:
            self._cum_progress = 0.0
            self._last_advance_step = self._step_count
        # waypoint 推进也重置（双向：前进时 target_idx 增大，倒车时减小）
        if n_pts > 0 and self._target_idx != self._prev_target_idx:
            self._last_advance_step = self._step_count
        self._prev_target_idx = self._target_idx

        # Stuck 检测仅在 penalty != 0 时启用。
        # penalty=0 时完全跳过——避免 episode 提前截断打断 RL 探索。
        stuck = False
        if self.reward_calc.stuck_penalty != 0:
            stuck_steps = self._step_count - getattr(self, "_last_advance_step", 0)
            stuck_window = getattr(self, "_stuck_window", 200)
            _end_margin = self._goal_end_wp_count
            near_path_end = (
                n_pts > _end_margin and self._target_idx >= n_pts - _end_margin
            )
            stuck = stuck_steps > stuck_window and not near_path_end

        if collision or goal_reached:
            reward, reward_comps = self.reward_calc.compute(
                done=True,
                collision=collision,
                goal_reached=goal_reached,
                goal_heading_error=_ghe,
                steps_taken=self._step_count,
                max_steps=self.max_steps,
                arc_completion=arc_completion,
                # B6 修复 (2026-08-19)：终止分支也必须传 stage，否则 RewardCalculator
                # 用默认 current_stage=0 → stage_scale 恒 0.2 → 碰撞只罚 -20。
                current_stage=max(self.curriculum_stage, 0),
            )

        terminated = collision or goal_reached
        timeout = self._step_count >= self.max_steps
        # ── 路径偏离判定（P2-17）────────────────────────────────────────
        # 用**推进窗口内**的 cte（_windowed_cte，与 _arc_position 同一投影口径）；
        # 没有可信投影时回退到 body 系全局 argmin 的 cte（旧行为）。阈值语义不变：
        # |cte| > path_deviation_limit（默认 4m）。
        # 旧实现直接吃全局 argmin 的 cte ⇒ 折返/迷宫平行走廊上另一条支路把 |cte|
        # 压低（实测车跑到 1m 外的支路上，|cte| 仍 0.0x）⇒ 4m 上限永不触发。
        cte_dev = cte if _cte_win is None else _cte_win
        path_deviation = abs(cte_dev) > self.path_deviation_limit
        truncated = timeout or path_deviation or stuck

        # ── 终止原因（用于诊断日志）──────────────────────────
        if collision:
            termination_reason = "collision"
        elif goal_reached:
            termination_reason = "goal_reached"
            self._eval_goal_count += 1
        elif timeout:
            termination_reason = "timeout"
        elif stuck:
            termination_reason = "stuck"
        elif path_deviation:
            termination_reason = "path_deviation"
        else:
            termination_reason = "none"

        # 截断原因的终止奖励（在 episode 结束时施加）
        # 路径偏离也作为终止惩罚单独处理，防止模型"学会更快偏离"
        if (timeout or stuck or path_deviation) and not collision and not goal_reached:
            reward, reward_comps = self.reward_calc.compute(
                done=True,
                timeout=timeout,
                stuck=stuck,
                path_deviation=path_deviation,
                arc_completion=arc_completion,
                # B6 修复：stuck / path_deviation 也乘 stage_scale（timeout 例外不乘）
                current_stage=max(self.curriculum_stage, 0),
                # 【2026-10-05 失败时刻补偿】必须把 steps_taken / max_steps 也转发过来。
                # reward.py 的 timeout 分支此前不接收这两个参数 ⇒ 恒取默认
                # (steps_taken=0, max_steps=500)，补偿项因 _steps_taken<=0 直接短路为 0，
                # 于是"早死更便宜"在 timeout 路径上完全没有被修掉（实测补偿恒等 0）。
                # 另两个分支（collision/goal）本就显式转发，此处补齐同一契约。
                steps_taken=self._step_count,
                max_steps=self.max_steps,
            )

        info = {
            "collision": collision,
            "goal_reached": goal_reached,
            "step": self._step_count,
            "cte": cte,  # body 系全局 argmin 的 cte（诊断用）
            "d_min": d_min,
            "progress": progress,
            "reverse_triggered": reverse_triggered,
            "safety_brake": safety_brake,
            "stuck": stuck,
            "timeout": timeout,
            "path_deviation": path_deviation,
            # 偏离判定**实际使用**的 cte（P2-17：推进窗口内的投影）。与上面的
            # 'cte' 不同源；解析一条 episode 为何被截断时看这一维。
            "path_deviation_cte": float(cte_dev),
            "termination_reason": termination_reason,
            # ── 极端回退标记（P2-20）：地图生成连续失败后回退成'空地+直线'，
            #    训练分布已与课程配置不符，必须让统计看得见。键**恒定存在**
            #    （False/True），否则 env_pool 的 info schema 校验会抛（首步冻结）。
            "map_fallback": bool(self._map_fallback),
            # ── 档位诊断 (2026-10-06)：path_dir 仍是 raw 几何真值（向后兼容），
            #    新增 path_dir_gate = 奖励闸门/偏航权重实际使用的滞后档位。
            "path_dir": path_dir,
            "path_dir_gate": path_dir_gate,
        }
        # ── 每步诊断字段（P2-22）────────────────────────────────────────
        # reward_components：训练路径**零消费者**（见 __init__ 的开关说明），
        #   实测 1711 B/条的 dict（7~15 个 float 键），可关。
        if self._emit_reward_components:
            info["reward_components"] = reward_comps
        # mapped_action：**始终保留**。它有真消费者（EnvStatsWrapper →
        #   delta_mean/v_mean → TB action/* 与 TUI，单位 rad、m/s），删掉会让
        #   诊断单位静默降级成 [-1,1]。按 action_dim 展开成 (float, ...) 元组而
        #   不是 ndarray.copy()：语义等价（np.array(list_of_tuples) 得到同样的
        #   (n, 2|3)），但每个元素省 ~198 B 对象开销。
        info["mapped_action"] = tuple(float(_v) for _v in mapped_action)
        # 动力学实际生效值（2026-11-04）：只在选了 limits/wheel 时新增该键 ——
        # none 档的 info 与加入本开关前逐位相同（键集合、值都不变）。
        if self._dynamics_level != "none":
            info["dynamics"] = dict(self._dynamics_info)
        # ── 可选传感器读数（默认全关 ⇒ 不新增任何键，信息路径零变化）──────
        # 与 reset() 共用 _sensor_info()：两处只有一套键名/结构口径。
        info.update(self._sensor_info())
        # 越界计数只在真的越界时出现（info 的键集合在合法回合里逐位不变）。
        if self._obs_clipped:
            info["obs_clipped"] = int(self._obs_clipped)

        return obs_np, reward, terminated, truncated, info

    def _get_raw_obs(self) -> dict:
        """桥的原始读数（雷达帧**不**在这里：它走 _lidar_frames()）。"""
        return {
            "odometry": self.bridge.get_odometry(),
            "path": self.bridge.get_path(),  # body frame，供 waypoint 推进使用
            "path_world": self.bridge.get_path_global(),  # world frame，供 PathExtractor 使用
        }

    def _append_privileged(self, obs_base: np.ndarray, odom: np.ndarray) -> np.ndarray:
        """为 Asymmetric Actor-Critic 追加 privileged 障碍物编码。

        Critic 可以看到这些额外信息；Actor 通过 ``obs[..., :input_dim]``
        将其切除。
        """
        robot_pose = (float(odom[0]), float(odom[1]), float(odom[2]))
        priv = self._priv_encoder.encode(robot_pose, self._obstacles)
        return np.concatenate([obs_base, priv])

    def reload_encoders(self, ckpt_path: str) -> None:
        """编码器已移入 Actor，环境侧无需加载编码器权重（向后兼容空操作）。

        该回调由 SubprocVectorEnv worker.send() 在恢复训练时触发。
        保留接口以便未来扩展（例如环境侧的辅助特征提取器）。
        """

    def update_curriculum(self, payload: bytes) -> None:
        """处理来自 trainer 的课程更新。

        由 Trainer._sync_curriculum_to_workers() 通过
        ``parent_remote.send(["setattr", ...])`` 推送 _pending_curriculum，
        reset() 入口检测并调用本方法。

        payload (pickle dict) 键:
            stage_config: dict            — 阶段地图配置（必填）
            defaults:     dict | None     — hybrid_astar 缺省值（可选）
            stage_index:  int | None      — 课程阶段索引（可选；B6 修复新增）。
                缺失（旧 payload）→ 保持原 stage_index 不变，向后兼容；
                None → trainer 无课程（不覆盖）。
        """
        import pickle

        # ── 防御：坏 payload 绝不能让 worker 崩掉 (2026-10-05) ──────────────
        # 实测事故：16 环境训练跑到 ~3 分钟时**全部 16 个 worker 在 reset 里崩死**，
        # 报 `_pickle.UnpicklingError: invalid load key, 'n'`
        # （/tmp/ackermann_worker_crash.log，21:19）。根因尚未定位：
        #   - 只有 trainer 会写这个键，且写的是 `pickle.dumps({...})`，类型检查也在；
        #   - pool 的 step/reset/set_curriculum 共用同一个 RLock ⇒ 不存在管道交错写；
        #   - 用真实配置 16 环境跑 256 步、以及 --resume 路径都未能复现。
        # 但无论根因是什么，"一条坏消息杀死整池 worker"都是不可接受的失败模式：
        # 课程 payload 丢失最坏只是这一段继续用旧 stage_config（与旧 payload 缺
        # stage_index 时的既有容忍策略一致），远好于训练直接起不来。
        try:
            data = pickle.loads(payload)
        except Exception as e:  # noqa: BLE001 - UnpicklingError/TypeError/EOFError…
            import sys as _sys

            _head = (
                bytes(payload[:24]) if isinstance(payload, (bytes, bytearray)) else b""
            )
            print(
                f"[WARN][curriculum] payload 无法反序列化（{type(e).__name__}: {e}）；"
                f"本段继续使用当前 stage_config。payload 类型={type(payload).__name__} "
                f"长度={len(payload) if hasattr(payload, '__len__') else '?'} 头部={_head!r}",
                file=_sys.stderr,
                flush=True,
            )
            return
        if not isinstance(data, dict):
            print(
                f"[WARN][curriculum] payload 解出 {type(data).__name__} 而非 dict，已忽略",
                file=__import__("sys").stderr,
                flush=True,
            )
            return
        stage_index = data.get("stage_index", None)
        if stage_index is not None:
            # env 侧权威值：reward stage_scale 依据 curriculum_stage 计算
            self._stage_index = int(stage_index)
        if self.map_generator is not None:
            self._sync_map_generator(
                data["stage_config"], data.get("defaults"), stage_index
            )

    def _sync_map_generator(self, stage_config, defaults, stage_index) -> None:
        """热更新 MapGenerator 配置（含可选的阶段索引）。

        stage_index 以关键字参数传入；envs/map_generator.py 的
        ``update_config(stage_config, defaults=None, stage_index=None)``
        仅在 kwarg 非 None 时更新其内部索引。
        为兼容旧签名（两参数版本）保留 try/except 回退——回退路径直接补写
        ``_stage_index``，保证 map_generator.stage_index 与 env 侧一致。
        """
        if stage_index is None:
            self.map_generator.update_config(stage_config, defaults)
            return
        try:
            self.map_generator.update_config(
                stage_config, defaults, stage_index=int(stage_index)
            )
        except TypeError:
            # 旧签名回退：两参数热更新 + 补写私有索引（property 无 setter）
            self.map_generator.update_config(stage_config, defaults)
            if hasattr(self.map_generator, "_stage_index"):
                self.map_generator._stage_index = int(stage_index)

    # ── 观测界 / 越界计数（3.0.0 W1）──────────────────────────────────
    def _bound_context(self) -> BoundContext:
        """把环境量摊成观测界的推导输入（构造期算一次）。"""
        xmin, xmax, ymin, ymax = (float(_v) for _v in self._world_bounds)
        lidar = {
            str(_spec.name): (float(_spec.range_min), float(_spec.max_range))
            for _spec in self._lidar_specs()
        }
        return BoundContext(
            dt=float(self.dt),
            v_max=float(self.v_max),
            v_min=float(self.v_min),
            vy_max=float(self.vy_max),
            omega_max=float(self.omega_max),
            lookahead=float(self._path_lookahead),
            world_diag=float(np.hypot(xmax - xmin, ymax - ymin)),
            lidar=lidar,
        )

    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        """把越界观测夹回 observation_space，并累加越界计数。

        只在真的越界时才夹：合法观测**原样返回同一个数组**（零拷贝，训练热路径
        不变）。NaN 既不计也不夹 —— 它是上游缺陷的信号，夹掉只会让缺陷更难查。
        计数累加在 self._obs_clipped，由 reset/step 在非零时写进 info。
        """
        low = self.observation_space.low
        high = self.observation_space.high
        with np.errstate(invalid="ignore"):
            out_of = (obs < low) | (obs > high)
        n = int(np.count_nonzero(out_of))
        if n == 0:
            return obs
        self._obs_clipped += n
        return np.clip(obs, low, high).astype(obs.dtype, copy=False)

    def _reset_info(self) -> dict[str, Any]:
        """reset 的 info：传感器读数 +（仅非零时）越界计数。"""
        info = dict(self._sensor_info())
        if self._obs_clipped:
            info["obs_clipped"] = int(self._obs_clipped)
        return info

    # ── 内建默认场景（3.0.0 W1）───────────────────────────────────────
    def _default_scenario_active(self, options: Mapping[str, Any] | None) -> bool:
        """本回合是否走 env 自带的默认场景。

        三条前提**全部**成立才接管；任一不成立都说明"世界由调用方负责"，env 不得
        把内置场景塞进去：

        1. options is None —— 调用方给了 options（含空 dict）就是显式声明场景由
           它定，保持 2.x 的直线兜底语义（W4b 的选项优先级不变）。
        2. self._bridge_owned —— 桥是 env 自己造的。注入的桥带着调用方自己的世界
           （路径、障碍、积分器），env 无从知道它的几何是否与 dense_static 兼容；
           把内置场景的 obstacles/path 递过去不是"补默认值"，而是拿别人的世界按
           env 的剧本演。PhysicalCommandBridge(NumpySimBridge) 那种注入形态恰好就是
           注入形态，且**静默**生效（桥照单全收，不报错）。
        3. self.map_generator is None —— 调用方给了 map_generator，地图由它生成；
           reset 的来源优先级已把 map_generator 排在默认场景之前，这里再挡一次
           是为了让判据自身自洽（读判据即可知道谁负责世界）。
        """
        return (
            options is None
            and self._bridge_owned
            and self.map_generator is None
            and bool(self._default_scenario_name)
        )

    def _roll_seed(self, seed: int | None) -> int:
        """默认场景本回合的重掷种子。

        显式 seed ⇒ 用它本身（同 seed 两次 reset 得到同一张图）；seed=None ⇒
        由 sensor_seed 派生并递增计数器（连续两次 reset 得到不同地图，同一
        sensor_seed 的两个 env 掷出同一串地图）。显式 seed 把计数器钉回 1。
        """
        if seed is not None:
            self._default_roll_index = 1
            return int(seed)
        roll = int(self.sensor_seed) * 1_000_003 + int(self._default_roll_index)
        self._default_roll_index += 1
        return roll

    def _default_scene(self, seed: int | None) -> dict[str, Any]:
        """默认场景的 reset options（场景 YAML 只加载一次）。"""
        if self._default_scenario_obj is None:
            from chassis_sim.scenarios import load_scenario

            self._default_scenario_obj = load_scenario(str(self._default_scenario_name))
        roll = self._roll_seed(seed)
        from chassis_sim.scenarios import roll_obstacles, scenario_to_reset_options

        options = dict(scenario_to_reset_options(self._default_scenario_obj, seed=roll))
        obstacles = list(options.get("obstacles") or [])
        options["obstacles"] = roll_obstacles(obstacles, roll)
        return options

    # ── 渲染（3.0.0 W1）─────────────────────────────────────────────
    def render(self):
        """按 render_mode 渲染一帧：rgb_array ⇒ uint8 (H, W, 3)；human ⇒ None。"""
        if self.render_mode is None:
            gym.logger.warn(
                "render() 被调用但 render_mode=None：请用 render_mode='human' 或 "
                "'rgb_array' 构造 env"
            )
            return None
        from chassis_sim.render import render_to_array

        pixels = render_to_array(self.bridge)
        if self.render_mode == "rgb_array":
            return pixels
        self._show_pixels(pixels)
        return None

    def _show_pixels(self, pixels: np.ndarray) -> None:
        """human 模式：贴到一张可复用 Figure 上并刷屏。

        非交互后端（Agg 等）只画不 pause —— plt.pause 在那种后端只会刷一条
        "non-GUI backend" 警告，不能显示任何东西。窗口图仍留在 _render_fig 上，
        调用方可自行 savefig。
        """
        import matplotlib
        import matplotlib.pyplot as plt

        if self._render_fig is None:
            self._render_fig, self._render_ax = plt.subplots()
        self._render_ax.clear()
        self._render_ax.imshow(pixels)
        self._render_ax.set_axis_off()
        if str(matplotlib.get_backend()).lower() not in _NON_INTERACTIVE_BACKENDS:
            plt.pause(0.001)

    def close(self):
        self._render_fig = None
        self._render_ax = None
        self.bridge.close()
