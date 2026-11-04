"""Ackermann 路径跟踪 DRL agent — 极简模型 v5 (2026-08-16 势能型重写)。

核心设计：
    1. 7 分量：cte (二次型) + heading + obstacle + progress (势能型) + steer + goal_proximity + step (时间税)；
       drive 已删——死代码
    2. 无动态权重、无布尔分支、无不可微点
    3. "定义目标，不教方法" — 删除所有硬编码驾驶规则

稠密分量（每步）：
    cte:             -w_cte × min(CTE², 9.0)   (二次型饱和, CTE≥3m 饱和→path_deviation 接管)
    heading:         -w_heading × |heading_dev| (防低速/倒车平行漂移，按 path_dir 预期 0/π)
    obstacle:        -w_obs × exp(-d/range) × speed_factor   (指数避障 × 速度感知放大)
                     2026-10-23：speed_factor = 1 + (|v|/v_max)**obstacle_speed_power，
                     obstacle_speed_power=0.0（默认）= 关闭 ⇒ 与旧公式逐位相同。
    progress:        w_progress × arc_progress  (势能型 2026-08-16: Φ(s_{t+1})−Φ(s_t)，前进+/后退−/振荡零)
    reverse:         -w_reverse × min(1, |v|/v_max)  **仅当 path_dir>0 且 v<0**（2026-10-29）
                     (前进段倒退的专项定价。arc_progress 按**路径比例**计量 ⇒
                      w_progress×arc_progress 在倒退时只有 −0.02~−0.04/步，
                      结构上罚不动"倒退直退到墙上"这类行为，故按车速单独定价)
    steer:           -w_steer × |prev_delta|    (转向平滑，防直线画龙)
    goal_proximity:  w_gp × ΔΦ, Φ(d)=1/(d+0.5)  (势能型 2026-08-19 修复 B7: 原 w_gp/(d+0.5) 恒定加成可刷分)
    step:            -0.02                      (微小时间税，抑制原地振荡)

终止分量（2026-10-23「失败经济学」重写，见下方的不变量推导）：
    collision:        collision_penalty × stage_scale − collision_surcharge
    goal:             goal_bonus + goal_heading_scale × cos(error) + goal_time_bonus
    timeout:          min(−timeout_min_penalty, timeout_penalty × stage_scale × (1 − ac))
    stuck:            -100.0 × stage_scale      （不变）
    path_deviation:   -100.0 × stage_scale      （不变）

【失败经济学不变量 (2026-10-23)】「撞墙」必须在**所有**课程阶段、所有进度上严格劣于
「超时」，否则策略会学会用撞车结束回合（论文实测：策略 180 集只超时 1 次，
而 DWA/MPC/PP/PP+avoid 超时占 25–60%）。旧实现的洞：
  ① timeout 不乘 stage_scale ⇒ timeout 单次惩罚在 stage0 是 collision 的 5 倍
     （−100 vs −20），于是"撞上去"比"走完超时"便宜；
  ② 两者都只在**单次**项上区分，量级远小于 300 步稠密累计（实测 stage2 撞车
     −106.8（62 步）vs 超时 −131.9（300 步）= 只差 25 分）。
只把 ① 修好（timeout 也乘 stage_scale）还会剩第二个洞：stage0 上撞墙 −20−S、
超时（ac≈0）−20·(1−ac) ⇒ 让"悬停到超时"的**净回报翻正**（补偿项最多 +refund_cap）。
所以最终设计 = 两处改动 + 一道下限：
  collision_terminal = collision_penalty · s − surcharge
  timeout_terminal   = min(−timeout_min_penalty, timeout_penalty · s · (1 − ac))
  其中 s = stage_scale ∈ [stage_scale_floor, 1] = [0.2, 1]、ac = arc_completion ∈ [0, 1]。
  surcharge **不随阶段缩放**（这正是它存在的理由：纯 stage_scale 项在 ac=0 时两侧
  完全相等，margin 退化成 0，没有任何"撞墙更差"的保证）。
  **margin 推导**（collision_penalty = timeout_penalty = −100）：
      margin ≡ timeout_terminal − collision_terminal = 100s + S + min(−P, −100s(1−ac))
      情形①  −100s(1−ac) ≤ −P  ⇒ margin = 100s·ac + S ≥ S
      情形②  −100s(1−ac) >  −P  ⇒ margin = 100s + S − P ≥ 100·0.2 + S − P
  默认 S = 60、P = 30 ⇒ margin ≥ min(S, 20 + S − P) = min(60, 50) = **50**，
  与 stage、与 ac 都无关地下界成立。走得越远（ac↑）超时越便宜 ⇒ margin 只增大。
  失败补偿项（_failure_time_comp）**不**削弱该不等式：collision 的 comp_scale = s，
  timeout 的 comp_scale = 1.0 ⇒ refund_collision = min(s·X, cap) ≤ min(X, cap)
  = refund_timeout（同一个上限）⇒ 补偿只会拉大 margin，不会吃掉它。
  下限 P 的存在理由与 S 不同：s 只到 0.2，若不设下限，低阶段的 timeout 基项会掉到
  −20·(1−ac)，配上 failure_dense_refund_cap = 120 就能让**长悬停超时回合净回报为正**
  （实测 +1.7）——失败回合净回报为正在本项目历史上每次都是 bug。

stage_scale = clip(current_stage / 3.0, stage_scale_floor, 1.0)  (2026-08-19 B6 修复: stage 索引必须由
env 端真实传入，此前恒为 -1 → current_stage=0 → 恒 0.2，碰撞只罚 -20);
下限 stage_scale_floor 默认 0.2（2026-10-23 由硬编码字面量改为可配，默认值不变）。
默认 S = 60、P = 30 时的终止基项（无失败补偿）:
    stage 0 → s=0.2000 → 碰撞 −100·0.2−60 = −80.00   超时 = min(−30, −20(1−ac)) = −30.00 (一切 ac)
    stage 1 → s=0.3333 → 碰撞 −93.33                 超时 −33.33 (ac=0) → −30.00 (ac≥0.10)
    stage 2 → s=0.6667 → 碰撞 −126.67                超时 −66.67 (ac=0) → −30.00 (ac≥0.55)
    stage 3 → s=1.0000 → 碰撞 −160.00                超时 −100.00 (ac=0) → −30.00 (ac≥0.70)
    stage 4/5 → s=1.0000 → 同 stage 3
  ⇒ **margin 最小 50 出现在 stage 0（超时锁在下限上）**；stage≥3 且 ac ≤ 0.7 时
     timeout 项与旧实现 −100×(1−ac) **逐位相同**；ac > 0.7 时下限把它从 −10…−30
     抬回 −30（有意为之：否则 stage3 走完 90% 再超时只罚 −10，抵不过 +120 的退款）。
"""

from __future__ import annotations

import numpy as np


class RewardCalculator:
    """极简 Reward v5：cte (二次型) + heading + obstacle + progress (势能型) + steer + goal_proximity + step。"""

    # ⚠ 本签名的数值默认值必须与 training/config.py 的 RewardConfig 字段默认、
    #   以及 config/reward.yaml 的 reward_params **三方逐位一致**
    #   （tests/test_config_default_sync.py::test_three_way_defaults_agree 锁住）。
    #   依据 P2-13：同一超参存在多套默认值时，任何绕过 YAML 的构造
    #   （RewardCalculator() / RewardConfig() / reward_cfg=None）都会静默跑一套
    #   从未标定过的奖励。2026-10-23 对齐到 YAML（2026-10-05 标定值）。
    #   obstacle_near_goal_* 原为 **kwargs 里的隐藏默认（0.0 / 2.0），不可内省、
    #   也无法被配置体系校验；P2-14 期间提为具名参数（默认值不变）。
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        obstacle_range: float = 0.8,
        collision_penalty: float = -100.0,
        # 撞墙的**不随阶段缩放**的附加罚 (2026-10-23「失败经济学」)：
        #     collision_terminal = collision_penalty × stage_scale − collision_surcharge
        # 它是把「撞墙严格劣于超时」从"有条件的"变成"恒成立的"那一项：
        #     margin = timeout_terminal − collision_terminal ≥ surcharge
        #            （精确式见模块 docstring；下界与 stage、与 ac 无关）
        # **默认 60.0**（标定：surcharge + 100·stage_scale_floor − timeout_min_penalty
        # = 60 + 20 − 30 = 50 是 margin 的全局下界）。0 = 关掉附加罚（A/B 回滚用，
        # 但"撞墙更便宜"的套利空间会一起回来）。
        collision_surcharge: float = 60.0,
        goal_bonus: float = 400.0,
        goal_heading_scale: float = 10.0,
        dense_scale: float = 1.0,
        stuck_penalty: float = 0.0,
        timeout_penalty: float = -100.0,
        # 超时基项的**下限**（2026-10-23 第二轮）：timeout 项从"不乘 stage_scale"
        # 改成"乘"之后，低阶段的基项会掉到 −20·(1−ac)，配上 failure_dense_refund_cap
        # = 120 足以让**长悬停超时回合的净回报为正**（实测 stage0 悬停 500 步 +1.7）。
        #     timeout_terminal = min(−timeout_min_penalty, timeout_penalty·s·(1−ac))
        # ⇒ 基项恒 ≤ −P（默认 −30），"悬停到超时"最坏也是 −30 − C(补偿) < 0。
        # 必须 > 0：0 等于关掉下限（会让上面的净回报翻正），<0 会把下限变成奖励。
        timeout_min_penalty: float = 30.0,
        path_deviation_penalty: float = -100.0,
        # stage_scale 的**下限**（失败惩罚随课程阶段放大的起步值，2026-10-23
        # 由硬编码 0.2 提为可配项）。默认 0.2 ⇒ 生产行为逐位不变，stage0
        # 碰撞仍是 -20.0。三向一致：本签名 / RewardConfig / reward.yaml。
        stage_scale_floor: float = 0.2,
        v_min: float = -0.6,
        delta_max: float = 0.526,
        goal_time_bonus: float = 0.0,
        step_penalty: float = -0.02,
        failure_time_compensation: float = 0.0,
        # 2026-10-23 诊断：0.0 → 0.12 / 0.0 → 120.0（与 RewardConfig 默认、
        # config/reward.yaml 同步；见 config/reward.yaml 该键的注释）。
        failure_dense_cap: float = 0.12,
        failure_dense_refund_cap: float = 120.0,
        obstacle_near_goal_factor: float = 0.0,
        obstacle_near_goal_range: float = 2.0,
        cte_range: float = 4.0,
        cte_delta: float = 0.5,
        # 速度感知障碍惩罚的幂次（2026-10-23）：见 compute() 的 r_obstacle 段。
        # 0.0 = 关闭（默认，逐位等于旧公式）；>0 = 距离相同时，车速越快罚得越狠
        # （factor = 1 + (|v|/v_max)**power ∈ [1, 2]）。
        # 上限不强制校验（因子本身已被 1+1=2 封顶，不会发散），但：
        # >2 会让满速端过度陡峭（d(factor)/du 在 u=1 处 = power）⇒ Q 在
        # a1≈+1 附近的梯度变陡，容易形成更硬的 bang-bang，除非做 A/B
        # 否则不建议超过 2~3。p 的真实作用见 reward.yaml 的
        # 每米成本判据 u^p < 2u−1（要"减速更划算"必须 p≥3）。
        obstacle_speed_power: float = 0.0,
        reverse_motion_penalty: float = 2.5,
        **kwargs,
    ):
        if obstacle_range <= 0:
            raise ValueError(f"obstacle_range must be > 0, got: {obstacle_range}")
        if dense_scale < 0:
            raise ValueError(f"dense_scale must be >= 0, got: {dense_scale}")
        if cte_range <= 0:
            raise ValueError(f"cte_range must be > 0, got: {cte_range}")
        # stage_scale 下限：必须落在 (0, 1] 内。0 会让 stage_scale 在低阶段为 0
        # ⇒ 碰撞/卡死/偏离的终止惩罚整项消失；>1 会让 clip 的上下界倒挂。
        if not (0.0 < float(stage_scale_floor) <= 1.0):
            raise ValueError(
                f"stage_scale_floor must satisfy 0 < x <= 1, got: {stage_scale_floor}"
            )
        # 撞墙附加罚必须 ≥ 0：负值会把"撞墙"变得比"超时"便宜——正是本参数要消除的
        # 套利方向（margin = surcharge + 100·s·ac ≥ surcharge ⇒ surcharge<0 时不再保证）。
        if float(collision_surcharge) < 0.0:
            raise ValueError(
                f"collision_surcharge must be >= 0 (0 = 关掉附加罚), "
                f"got: {collision_surcharge}"
            )
        # 超时下限必须 > 0：它把 timeout 基项钉在 −P 以内。0 会让低阶段的
        # "悬停到超时"净回报翻正（见模块 docstring），负值更是把下限变成奖励。
        if not (float(timeout_min_penalty) > 0.0):
            raise ValueError(
                f"timeout_min_penalty must be > 0 (0 would disable the floor), "
                f"got: {timeout_min_penalty}"
            )
        self.weights = weights or {
            "drive": 7.0,
            "cte_penalty": 2.0,
            "heading_penalty": 0.3,
            "heading_penalty_reverse": 1.0,
            "obstacle": 1.0,
            "collision": 1.0,
            "goal": 1.0,
            "goal_heading": 1.0,
            "alive": 0.0,
            "stuck": 1.0,
            "timeout": 1.0,
            "path_deviation": 1.0,
            "steer_penalty": 0.1,  # 转向惩罚——防直线画龙
            "goal_proximity": 0.0,  # 终点距离势能权重 (势能型 2026-08-19)，默认关闭
            # progress 此前只以 w.get('progress', 4.0) 的字面量存在于 compute()，
            # 不在本字典 ⇒ 同一超参两套默认值（P2-13 类问题）。补进字典：值与原
            # 字面量一致，weights=None 的调用点行为逐位不变。
            "progress": 4.0,
        }
        # near_goal 内的 obstacle 项门控（P2-14）：原为 **kwargs 隐藏默认（0.0 / 2.0），
        # 无法从 RewardConfig / reward.yaml 配置 ⇒ 行为被硬钉在"终点附近整项清零"。
        # 现为具名参数 + 配置白名单 + reward.yaml 键；**默认值未变**（0.0 = 清零）。
        if obstacle_near_goal_factor < 0:
            raise ValueError(
                f"obstacle_near_goal_factor must be >= 0 (0 = close obstacle term near goal), "
                f"got: {obstacle_near_goal_factor}"
            )
        if obstacle_near_goal_range <= 0:
            raise ValueError(
                f"obstacle_near_goal_range must be > 0, got: {obstacle_near_goal_range}"
            )
        self.obstacle_near_goal_factor = float(obstacle_near_goal_factor)
        self.obstacle_near_goal_range = float(obstacle_near_goal_range)
        # 私有别名保留：此前这两个值只以私有属性存在，env 侧接线测试
        # （tests/test_x6_env_policy.py::TestObstacleNearGoalForwarding）按旧名读取。
        self._obstacle_near_goal_factor = self.obstacle_near_goal_factor
        self._obstacle_near_goal_range = self.obstacle_near_goal_range
        self.obstacle_range = obstacle_range
        self.collision_penalty = collision_penalty
        # 撞墙的附加罚（不随 stage_scale 缩放）：见 compute() 的 collision 分支。
        self.collision_surcharge = float(collision_surcharge)
        self.goal_bonus = goal_bonus
        self.goal_heading_scale = goal_heading_scale
        self.dense_scale = dense_scale
        self.stuck_penalty = stuck_penalty
        self.timeout_penalty = timeout_penalty
        # 超时基项下限（见 compute() 的 timeout 分支）。
        self.timeout_min_penalty = float(timeout_min_penalty)
        self.path_deviation_penalty = path_deviation_penalty
        # stage_scale 的下限（见 compute()）。历史值硬编码 0.2，现可配，默认不变。
        self.stage_scale_floor = float(stage_scale_floor)
        self.delta_max = abs(delta_max) if delta_max != 0 else 0.526
        self.goal_time_bonus = goal_time_bonus
        self.step_penalty = step_penalty
        # 时间税补回倍率 (2026-10-05)：见 compute() 的 _failure_time_comp。
        # 默认 0.0 = 关闭——实测该补偿方向与目标相反（把"早失败少付的时间税"
        # 再罚一遍，25/120 步失败差反而 7.5 → 9.8）。保留为可调项，便于 A/B。
        self.failure_time_compensation = float(
            kwargs.get("failure_time_compensation", failure_time_compensation)
        )
        # 失败回合"稠密累计量封顶"**速率** (2026-10-05)：上限 C = 该值 × max_steps，
        # |Σ稠密| 超过 C 的部分在终止时退回，使失败代价不再随步数线性增长。
        # 取"速率×步数预算"而不是绝对值：补偿量必须随回合预算缩放，否则
        # max_steps 变小/变大时这一步会失衡（例如 max_steps=500 时 C=60，
        # max_steps=150 时 C=18）。0 = 关闭该补偿项。
        self.failure_dense_cap = float(
            kwargs.get("failure_dense_cap", failure_dense_cap)
        )
        # 失败补偿的**单次退款上限**（见 _failure_time_comp ③）：不设上限时
        # 终止步会出现 +3562 量级的尖峰，把"长失败回路的前一步"变成价值最高状态。
        self.failure_dense_refund_cap = float(
            kwargs.get("failure_dense_refund_cap", failure_dense_refund_cap)
        )
        # 本回合稠密分累计（供失败时刻补偿，见 compute 末尾）
        self._dense_sum = 0.0
        self._dense_steps = 0
        # 速度感知障碍惩罚的幂次：必须 ≥ 0。0 = 关闭（默认）——
        # 关闭时 compute() 里整段跳过该因子（不乘 1.0），保证旧路径逐位不变。
        # 上限不强制（>2 会过度放大高速罚分，见 compute() 的注释）。
        if float(obstacle_speed_power) < 0:
            raise ValueError(
                f"obstacle_speed_power must be >= 0 (0 = disabled), "
                f"got: {obstacle_speed_power}"
            )
        self.obstacle_speed_power = float(obstacle_speed_power)
        # 前进段倒退的专项定价（2026-10-29）：必须 ≥ 0。默认 2.5 = **开启**
        # （与 reward.yaml / RewardConfig 同值，三向一致是 P2-13 的硬约束）；
        # 0 = 关闭 ⇒ compute() 里整段跳过，回到 2026-10-29 之前的逐位行为。
        if float(reverse_motion_penalty) < 0:
            raise ValueError(
                f"reverse_motion_penalty must be >= 0 (0 = disabled), "
                f"got: {reverse_motion_penalty}"
            )
        self.reverse_motion_penalty = float(reverse_motion_penalty)
        self.cte_range = float(cte_range)
        self.cte_delta = float(cte_delta)
        self._cte_max = (
            3.0  # CTE 饱和上限 (m)——正常驾驶 CTE<3m 全程有梯度，饱和仅在极端避障触发
        )

    def reset(self) -> None:
        """回合边界钩子：清空本回合稠密累计。

        compute() 末尾原本只靠"steps_taken 严格回退"猜回合边界（见那里的注释）：
        该启发式依赖 env 每步恰好一次稠密调用、终止调用带 n+1 —— 任何重构（重构
        调用点、并行采样、某步漏调一次）打破它就跨回合污染 _dense_sum，失败补偿
        静默错乱。AckermannEnv.reset() 现在显式调用本方法，启发式退化为兜底。
        """
        self._dense_sum = 0.0
        self._dense_steps = 0

    def compute(
        self,
        done: bool = False,
        collision: bool = False,
        goal_reached: bool = False,
        timeout: bool = False,
        cte: float = 0.0,
        heading_error: float = 0.0,
        d_min: float = 10.0,
        progress: float = 0.0,
        v: float = 0.0,
        v_max: float = 2.0,
        prev_delta: float = 0.0,
        prev_v: float = 0.0,
        delta: float = 0.0,
        reverse_triggered: bool = False,
        goal_heading_error: float = 0.0,
        stuck: bool = False,
        path_deviation: bool = False,
        near_goal: bool = False,
        curvature: float = 0.0,
        target_curve: float = 0.0,
        direction_change: bool = False,
        displacement: float = 0.0,
        steps_taken: int = 0,
        max_steps: int = 500,
        initial_dist: float = 0.0,
        current_dist: float = 0.0,
        arc_completion: float = 0.0,
        current_stage: int = 0,
        **kwargs,
    ) -> tuple[float, dict[str, float]]:
        # --- 终止奖励 ------------------------------------------------
        # 阶段缩放：碰撞/卡死/偏离统一缩放，消除"懦夫策略"套利空间。
        # stage 0→scale=0.2: 撞墙=-20, 卡死=-20, 偏离=-20——都是 -20，
        # 但不动每步 -0.05×100=-5，累计 stuck=-20 → 总 -25，
        # 开 50 步撞墙：-2.5+10-20=-12.5 → 优于不动。鼓励探索。
        # stage 3+→scale=1.0: 全额定惩罚。
        # 超时 (2026-08-16 v5 按未完成度缩放；2026-10-23 起**再乘 stage_scale**，并有绝对下限):
        #     min(−timeout_min_penalty, −100 × stage_scale × (1 − arc_completion))
        # 2026-10-23 之前它不乘 stage_scale，于是「撞墙(×0.2)」比「超时(×1.0)」便宜 5 倍，
        # 策略学会用撞车结束回合（论文实测：策略 180 集只超时 1 次）。
        # 撞墙侧同时加一项不随 stage 缩放的 collision_surcharge（默认 60），
        # 使 margin = timeout − collision ≥ collision_surcharge = 60 > 0 恒成立
        # （下界在 stage0 且下限未触发时取到：−30 − (−20 − 60) = 50）。
        # 【2026-08-19 B6 修复】stage_scale 只依赖 current_stage，而 current_stage 来自
        # AckermannEnv.curriculum_stage；此前 MapGenerator 未收到 stage_index，
        # 该值恒为 -1 → max(-1,0)=0 → scale 恒 0.2（碰撞只 -20，intended -66.7/-100）。
        # 现由 make_env(stage_index=...) 端到端注入，update_curriculum payload 亦可热更新。
        # 下限 self.stage_scale_floor（默认 0.2 = 历史硬编码值，逐位不变）。
        stage_scale = float(np.clip(current_stage / 3.0, self.stage_scale_floor, 1.0))

        # ── 失败时刻补偿 (2026-10-05)：消除"早死更便宜" ──────────────────
        # 结构性缺陷：稠密分按步累计（无上限），而终止惩罚封顶 -100×stage_scale。
        # 同一个失败 25 步发生 = -205，116 步发生 = -587（差 382），于是
        # "抄近道/自杀"（几十步内撞完）在 Q 上优于"认真沿路径执行到失败"。
        #
        # 修法（组合拳，全部在终止分支、全部有界）：
        #   ① （已实测否决）时间税补回 comp1 = -|step_penalty| × steps_taken：
        #      实测反而把 25/120 步失败差从 7.5 拉大到 9.8 —— 因为它把"早失败
        #      少付的时间税"**再罚一遍**（25 步罚 -0.50、120 步罚 -2.32，
        #      后者被罚得更狠），方向与目标相反。用户原方案 (max_steps−steps)
        #      ×|step_penalty| 在 max_steps=500 时恒为 -9.5/-10，与发生时刻
        #      几乎无关（等于没修）。两条路都不解决问题，故默认关闭
        #      (failure_time_compensation=0.0)；时间税本身量级只有 0.02/步，
        #      从来不是"早死便宜"的主因。
        #   ② 稠密累计量封顶：失败时把本回合累计稠密分 |Σdense| 超过 C 的部分
        #      退回去（C = failure_dense_cap × max_steps，默认速率 0.12 ⇒ C=60 @500 步）：
        #        comp2 = +max(0, |Σdense| − C)   （仅当 Σdense < 0）
        #      作用是让"失败代价"不再随步数线性增长——稠密分按步累计是
        #      "早死更便宜"的主因，封顶把它从 O(steps) 压到 O(C) 有界。
        #      这是**终端再平衡**：per-step 稠密分一个都没改，梯度形状不变，
        #      只是把"单回合稠密缺陷"的总额收敛到同一量级（每回合一次性的
        #      终止调整，不是逐步奖励，故不会改变策略梯度在轨迹上的形状）。
        # 实测结论：失败回合的总回报 = 终止基项 − clip(|Σdense|, 0, C) + O(步长)：
        #   - 对 |Σdense| ≥ C 的失败，代价被钉在 终止基项 − C（与发生时刻无关）；
        #   - 对 |Σdense| < C 的失败，代价仍随缺陷单调（不会出现"越差越好"）。
        # 实测（解析核算 + 真 stage4 地图，见 /tmp/rw_verify_final.py 与 /tmp/rw_ref.py）：
        #   · 稠密速率 −4.18/步（用户实测数字）、max_steps=500、碰撞终止：
        #     25 步失败 vs 116 步失败 —— 旧 |Δ|=382.2 → 新 |Δ|=1.8（C=60）；
        #     绝对值 −160.5 / −162.3，量纲未变。
        #   · 真 stage4 地图（带障碍，6 次 oracle）：失败回合 旧 −108/−133/−561/
        #     −182/−380 → 新 −108/−133/−160/−160/−160（有界）；到达终点那次
        #     旧 +75.1 → 新 +53.6（仍为唯一正项，相对优势保持）。
        # 量纲：|comp2| = |Σdense| − C ≤ |Σdense|（本就被稠密分界住），
        # |comp1| ≤ 时间税满额 ⇒ 单次终止调整 ≤ |Σdense| 量级，
        # per-step 奖励完全不变，总回报被界在 终止基项 − C 附近
        # （真 stage4：−160；旧行为无界，实测低到 −561）。
        # 成功分支（goal）**不加**任何补偿——必须保持成功回合的相对优势。
        # 两个补偿项都可独立关闭：failure_time_compensation=0.0 /
        # failure_dense_cap=0.0（两者都设 0 即完全退回旧行为，便于 A/B 回滚）。
        # 调参入口：failure_dense_cap 是"每步额度"——调大 = 失败代价更负、
        # 补偿更温和；调小 = 失败代价更平（更接近"与时刻无关"）。
        # 【2026-10-05 已接线】两个键都已进 RewardConfig 白名单（training/config.py）
        # 并由 ackermann_env 转发 ⇒ **改 config/reward.yaml 的 reward_params 即可**，
        # 不再需要从构造函数覆盖（有行为级测试：tests/test_reward_calibration.py）。
        _steps_taken = int(steps_taken) if steps_taken else 0

        def _failure_time_comp(comp_scale: float) -> float:
            """失败回合的"没走完的时间 + 稠密累计封顶"补扣；成功分支不调用。"""
            if comp_scale <= 0.0 or _steps_taken <= 0:
                return 0.0
            _comp = 0.0
            # ① 时间税补回
            if self.failure_time_compensation > 0.0:
                _comp -= (
                    comp_scale
                    * self.failure_time_compensation
                    * abs(float(self.step_penalty))
                    * float(_steps_taken)
                )
            # ② 稠密累计量封顶（只退"超额"部分，不改变每步奖励）
            _cap = float(self.failure_dense_cap) * float(max(int(max_steps), 0))
            if _cap > 0.0:
                _sum = float(self._dense_sum)
                if _sum < -_cap:
                    _refund = comp_scale * (-_sum - _cap)
                    # ③ 退款上限 (2026-10-05)：② 是**终止步的一次性大额退款**。
                    # 实测 stage5 的终止奖励高达 +3562 / +2183 / +1851（500 步
                    # timeout），是 goal_bonus(400) 的 8.9× ⇒ 它把"长失败回路的
                    # 前一步"变成 MDP 里**价值最高的状态**（缩放后 +534），
                    # 并让 4.2% 的 target 撞上 +150 上限（见 CLAUDE.md「Critic 稳定性」）。
                    # 加上限后：失败代价依然有界（终止基项 − C），终止尖峰 ≤ cap，
                    # target 不再被截断。代价是 |Σdense| > C+cap 的极长失败会重新
                    # 出现长度依赖——这是可接受的，因为那种回合本就该更差。
                    # 取值 500 是同时满足两个目标的区间下沿：主目标「25 步失败 ≈ 116 步失败」
                    # 在稠密速率 4.18/步、116 步时需要退满 485−60=425，故上限必须 ≥425
                    # （150 时该性质被破坏，spread 1.8→277）；而终止尖峰 = −100+cap ≤ +400，
                    # 乘 reward_scale 0.15 后 ≤60，远低于 target_q 上限 150。
                    # cap <= 0 ⇒ 不限（完全回退旧行为，便于 A/B 回滚）。
                    # 【2026-10-23】生产值改为 120（config/reward.yaml），理由是 ≤ +120 的
                    # 终止尖峰（×0.15 = +18）已远离 target_q 的 150 截断。代价必须知道：
                    #   · min() 作用在**已乘 comp_scale** 的 _refund 上 ⇒ 上限不随 stage_scale
                    #     缩放；stage0 长失败碰撞的终止奖励上界因此是 −20+120 = **+100**。
                    #   · |Σdense| > C+cap 的失败重新有长度依赖（实测 stage0 30 步碰撞
                    #     −29.6 vs 300 步超时 −217.2，仍差 7.3 倍，见 /tmp/verify_failure_cost.py）。
                    # 要真正抹平长度依赖需要更大的 cap，但那会把终止步推成正奖励（历史上
                    # 就是这个 bug），所以 120 是"尖峰可控"与"长度依赖收敛"之间的折中。
                    if self.failure_dense_refund_cap > 0.0:
                        _refund = min(_refund, self.failure_dense_refund_cap)
                    _comp += _refund
            return _comp

        if done and collision:
            # 撞墙 = 单次惩罚 × stage_scale − **不随 stage 缩放**的附加罚 (2026-10-23)。
            # 附加罚存在的唯一理由：让「撞墙 < 超时」与 stage、与 ac 都无关地成立。
            # 单靠 collision_penalty × stage_scale 做不到——timeout 项也乘 stage_scale 后
            # 两者在 ac=0 时只差 0（−100s vs −100s），margin 会退化成 0；加上 surcharge
            # 后 margin = surcharge + 100·s·ac ≥ surcharge 恒成立（推导见模块 docstring）。
            _total = (
                self.weights.get("collision", 1.0)
                * self.collision_penalty
                * stage_scale
                - self.collision_surcharge
            )
            _total = _total + _failure_time_comp(stage_scale)
            return float(_total), _zero_comps("collision", _total)

        if done and goal_reached:
            r_goal_heading = np.cos(goal_heading_error)
            r_goal_base = self.goal_bonus
            r_gh = (
                self.weights.get("goal_heading", 0.0)
                * r_goal_heading
                * self.goal_heading_scale
            )
            r_goal_total = self.weights.get("goal", 1.0) * r_goal_base
            # 到达时间奖金：越早完成奖金越高
            if self.goal_time_bonus > 0 and max_steps > 0:
                r_goal_time = self.goal_time_bonus * (
                    1.0 - min(steps_taken, max_steps) / max_steps
                )
            else:
                r_goal_time = 0.0
            total = r_goal_total + r_gh + r_goal_time
            comps = _zero_comps("goal", total)
            comps["goal"] = float(r_goal_total)
            comps["goal_heading"] = float(r_gh)
            comps["goal_time"] = float(r_goal_time)
            return float(total), comps

        if done and timeout:
            # 按未完成度惩罚，**从 2026-10-23 起同样乘 stage_scale 并带下限**：
            #     timeout_terminal = min(−timeout_min_penalty,
            #                            timeout_penalty × stage_scale × (1 − ac))
            # 两层各自修一个洞：
            #   · 乘 stage_scale —— 旧实现不乘（2026-08-16 v5 的"例外"），后果是 stage0 的
            #     超时单次罚是撞墙的 5 倍（−100 vs −20）⇒ 撞上去反而更便宜；
            #   · 下限 −P —— 只乘 stage_scale 时低阶段基项掉到 −20·(1−ac)，配 120 的
            #     退款能让长悬停超时回合净回报翻正（实测 +1.7）。
            # stage_scale = 1.0（stage≥3）且 ac ≤ 1 − P/100 = 0.7 时本项与旧实现
            # −100×(1−ac) **逐位相同**；ac > 0.7 时被下限抬到 −P。
            ac = float(np.clip(arc_completion, 0.0, 1.0))
            total = min(
                -self.timeout_min_penalty,
                self.weights.get("timeout", 1.0)
                * self.timeout_penalty
                * stage_scale
                * (1.0 - ac),
            )
            # 补偿的 comp_scale 仍取 1.0（**退款语义不动**）：stage_scale ≤ 1 ⇒
            # refund_collision = min(s·X, cap) ≤ min(X, cap) = refund_timeout
            # ⇒ 补偿只会把 margin 拉得更大，不会吃掉 surcharge 保证的那 40。
            total = total + _failure_time_comp(1.0)
            return float(total), _zero_comps("timeout", total)

        if done and stuck:
            _total = self.weights.get("stuck", 1.0) * self.stuck_penalty * stage_scale
            _total = _total + _failure_time_comp(stage_scale)
            return float(_total), _zero_comps("stuck", _total)

        if done and path_deviation:
            _total = (
                self.weights.get("path_deviation", 1.0)
                * self.path_deviation_penalty
                * stage_scale
            )
            _total = _total + _failure_time_comp(stage_scale)
            return float(_total), _zero_comps("path_deviation", _total)

        # ── 稠密分量 (2026-07-18 v2: 3 分量，纯线性) ──────────────────────
        s = self.dense_scale
        w = self.weights

        heading_error_local = float(kwargs.get("heading_error_local", heading_error))

        # ── 1. r_cte: 二次型跨轨惩罚 (带饱和) ──────────────────────────
        # 近路径梯度≈0 防震荡，远路径饱和防避障时过度惩罚。
        # CTE > cte_max 时 penalty 封顶——path_deviation 才是主信号。
        _cte_sq = min(cte**2, self._cte_max**2)
        r_cte = -w.get("cte_penalty", 2.0) * _cte_sq * s

        # ── 3. r_heading: 偏航惩罚 — 防低速/倒车平行漂移 (2026-10-05 修复) ──
        # 【修复】"预期朝向"恒为 0，不再按 path_dir 取 0/π。
        # 依据：上下文的 heading_error_local = 路径鼻向 − 车头朝向，而路径第 2 列
        # 存的是**鼻向**（RS 积分，见 ackermann_env.py 与 map_generator.py 的注释）。
        # 因此正确跟踪时**前进段与倒车段它都 ≈0**，只有"该倒车却掉头正着开"才 ≈±π。
        # 原实现 `_expected = 0 if path_dir>=0 else π`，而 path_dir 又由
        # heading_error_local 反推 ⇒ 代入后恒等变形成 `min(|h|, π−|h|)`：
        # 方向盲 —— "倒车段掉头正着开"零惩罚，Q 上与真倒车完全等价。
        # 改成恒取 0 后，该项才真正能做到它注释里声称的事：
        #   正确前进/倒车 → h≈0 → 不罚；  该倒车却正着开 → h≈±π → 罚满。
        # 注意惩罚幅值上限由 π/2·w 变为 π·w（0.3→0.94/步），这正是"方向约束"
        # 应有的量级；path_dir 仍由调用方传入（供其它诊断/日志使用），此处不再使用。
        _heading_dev = abs((heading_error_local + np.pi) % (2 * np.pi) - np.pi)
        # 【2026-10-05】倒车段用**更大的**权重：让"掉头正着开"明显不划算。
        # 前进段仍是 0.3（|h| 上限 π ⇒ 最大 -0.94/步），倒车段提到 1.0
        # （最大 -3.14/步）。配合 env 侧的档位闸门（反向前进不给正向 arc_progress），
        # 两者叠加后"正确倒车"与"掉头正着开"的单步差距 ≈ 3.4：
        #   正确: heading 0 + progress +0.30 = +0.30
        #   作弊: heading -3.14 + progress 0   = -3.14
        _w_head = w.get("heading_penalty", 0.3)
        if float(kwargs.get("path_dir", 1.0)) < 0:
            _w_head = w.get("heading_penalty_reverse", 1.0)
        r_heading = -_w_head * _heading_dev * s

        # ── 4. r_obstacle: 指数避障 ─────────────────────────────────────
        obstacle_range = max(self.obstacle_range, 0.01)
        r_obstacle = -w.get("obstacle", 1.0) * np.exp(-d_min / obstacle_range) * s
        # ── 4b. 速度感知放大（2026-10-23，**默认关闭**）──────────────────────
        # 动机：上面这行只看距离 ⇒ 同一个位置上"全速冲过"与"减速蹭过"的即时
        # 罚分**完全相同**，critic 的价格信号里没有任何"减速"的经济理由
        # （Q 校准审计：92.9% 的状态 argmax 落在边界 a1=+1.0 最大速度）。
        # 开启后（power>0）：
        #     factor = 1 + (|v|/v_max) ** power        ∈ [1, 2]
        #       |v| = 0      → factor = 1（静止时与旧值逐位相同）
        #       |v| = v_max  → factor = 2（满速时罚分翻倍）
        # 用 |v| 而不是 v：倒车贴近障碍同样该罚（档位与风险无关）。
        # v 是 env 侧传进来的**当步实际执行车速**（ackermann_env.py 的
        # v = raw['odometry'][3]，经加速度限幅之后的值），v_max 取 env 配置。
        # 裁 |v|/v_max ≤ 1：factor 上界锁死在 2，超速/回绕不放大上界。
        # power = 0 时**整段跳过**（不乘 1.0），生产行为逐位不变。
        # ⚠ 幂次的真实作用是"把满速端拉陡"：factor(u)=1+u^p，u=1 处恒为 2，
        #   而 u<1 时 p 越大 factor 越接近 1（u=0.7：p=1→1.70、p=3→1.34）。
        #   因此 p 越大，"略低于满速"相对"满速"的罚分优势越明显 —— 这正是
        #   "该减速"的激励可能出现的位置；p 越小则只是整体放大、几乎不动动作排序。
        #   定量判据见 config/reward.yaml 的解释与 tests/test_obstacle_speed_power.py。
        if self.obstacle_speed_power > 0:
            _vmax_abs = abs(float(v_max))
            if _vmax_abs > 1e-9:
                _speed_ratio = min(abs(float(v)) / _vmax_abs, 1.0)
                r_obstacle *= 1.0 + _speed_ratio**self.obstacle_speed_power
        # ── near_goal 门控（P2-14：语义已在 2026-10-23 核实并写清）─────────
        # near_goal 由 env 侧给出（距终点 < 2m，见 ackermann_env.py），作用**仅限**
        # obstacle 这一项，其它分量一律照算。两条分支：
        #   factor == 0（默认，与 2026-10-05 以来的行为逐位一致）
        #       ⇒ r_obstacle 置 0.0：**整项清零**，不是"衰减到 0"，也不留残值；
        #         实测终点前一步 obstacle 从 −1.3237 直接变 0.0。
        #          语义 = "终点附近撞不撞无所谓"（有意为之还是遗留：代码未留理由，
        #          故本轮只把它变成可配，不动默认行为）。
        #   factor > 0
        #       ⇒ r_obstacle *= max(factor, goal_dist / obstacle_near_goal_range)：
        #         以 factor 为**下限**的线性衰减——越靠近终点折扣越狠，
        #         但永远不会低于 factor×原值（不会完全归零）。
        # 两个参数现由 config/reward.yaml 的 reward_params 配置，经 RewardConfig
        # 白名单转发到 RewardCalculator。
        if near_goal:
            if self.obstacle_near_goal_factor > 0:
                # 线性衰减：距终点越近 obstacle 权重越低，但不低于 factor（不归零）
                goal_dist = float(kwargs.get("goal_dist", 0.0))
                decay = max(
                    self.obstacle_near_goal_factor,
                    goal_dist / max(self.obstacle_near_goal_range, 0.01),
                )
                r_obstacle *= decay
            else:
                r_obstacle = 0.0  # factor=0：整项清零（默认行为，未改变）

        # ── 5. r_progress: 势能型沿路径位移 (2026-08-16 v5) ─────────────
        # arc_progress = Φ(s_{t+1}) − Φ(s_t) = w × (arc_pos_cur − arc_pos_prev)
        # 前进为正、后退为负、振荡净零——结构性消灭"终点前徘徊"套利。
        # 势能总量有界 = w × 路径总长；沿路径走完全程累计势能奖励 = w × 总长
        # （goal 步的稠密奖励被终止 compute 覆盖，由终止奖励接管）。
        arc_progress = float(kwargs.get("arc_progress", progress))
        r_progress = w.get("progress", 4.0) * arc_progress * s

        # ── 5c. r_reverse: 前进段倒退的专项定价 (2026-10-29) ──────────────
        # 病（run110 最后 20 万步实测，stage=obstacles_dense）：起点 27~37% 的确定性
        #   动作给出 v<0；沿路径倒退每步只让 arc_progress 掉 ~0.006（它是路径**比例**）
        #   ⇒ 4.0×arc_progress ≈ −0.02~−0.04/步，几乎免费 ⇒ 「倒退直退到边界墙」
        #   全程无成本（预览 step_990080：21 步退 1.9 m 撞墙，总回报 −144.8
        #   = 撞墙终局 −126.7 + 稠密 −18.1）。
        # 判据只用两侧真实量：path_dir（env 侧滞后稳定的档位）+ v（当步执行车速）。
        #   · 只罚 path_dir>0（路径要求前进）且 v<0（实际倒退）⇒ 倒车段上正确倒车
        #     **不被罚**；"倒车段掉头正着开"仍由 heading_penalty_reverse 负责。
        #   · min(1, |v|/v_max) 线性归一 ⇒ 满速倒退拿满 w_reverse，微退只有零头
        #     （位姿微调倒车仍可用，不锁死倒挡）。
        r_reverse = 0.0
        if (
            self.reverse_motion_penalty > 0.0
            and float(kwargs.get("path_dir", 1.0)) > 0.0
            and float(v) < 0.0
        ):
            r_reverse = (
                -self.reverse_motion_penalty
                * min(1.0, -float(v) / max(float(v_max), 1e-6))
                * s
            )

        # ── 5b. r_steer: 转向平滑惩罚——防直线画龙 ─────────────
        # 【2026-10-05 修复死分量】原实现写 kwargs.get('prev_delta', 0.0)，
        # 但 prev_delta 是 compute() 的**命名参数**（见上方签名），调用方无论
        # 怎么传（env 侧 ackermann_env.py:642 一直是 prev_delta=float(prev_delta)），
        # 它都只会绑定到命名参数 ⇒ kwargs 里**永远没有这个键** ⇒ 该项恒为 0。
        # 实测：rc.compute(prev_delta=0.526) 与 prev_delta=0.0 返回逐位相同
        # （total=-0.073511, steer=-0.000000）；6 阶段 3982 步专家轨迹 steer
        # 均值 0.000。"防直线画龙"从未生效过（本应最多 -0.1×0.526=-0.0526/步）。
        # 修法：直接用命名参数 prev_delta。
        _prev_delta = float(prev_delta)
        _w_steer = w.get("steer_penalty", 0.0)
        r_steer = -_w_steer * abs(_prev_delta) * s if _w_steer > 0 else 0.0

        # ── 6. r_goal_prox: 终点距离势能 (2026-08-19 势能化，修 B7 套利) ──────
        # 旧形式 r = w/(d+0.5) 是"恒定稠密加成"（非势能），不随动作变化：
        # 在 1.5m 处悬停 400 步 = +400 稠密奖励，远超到达终点收益 → 可无限刷分。
        # 新形式为势能差分（与 arc_progress 同构）：
        #     r = w × [Φ(d_t) − Φ(d_{t−1})],  Φ(d) = 1/(d+0.5)
        # 望远镜求和有界：Σ r = w×[Φ(d_T) − Φ(d_0)] ≤ w×[1/0.5 − 1/(d_0+0.5)] ≤ 4.0 (w=2.0)
        #   - 2m 内势能差上限 = w×(1/0.5 − 1/2.5) = 2.0×1.6 = 3.2
        #   - 逼近为正、远离为负、恒距悬停每步 ≈ 0 → 结构性消灭"终点前徘徊"套利
        # 有意不再用 near_goal 门控：门控对差分形式不连续（跨越 2m 边界时势能差包含
        # 势能本身的跳变），且"入场即得 w×Φ(2.0)=1.6"的边界尖峰会造成价值函数断层。
        # Φ 随 d 自然衰减（d=2m 时 w×|Φ'| = 0.32/m，仅为 arc_progress 4.0/m 的 8%），
        # 远场影响可忽略，无需门控。near_goal 仍用于 obstacle 关闭逻辑（见上）。
        _w_gp = w.get("goal_proximity", 0.0)
        if _w_gp > 0:
            _gd = float(kwargs.get("goal_dist", 10.0))
            _gd_prev = kwargs.get("goal_dist_prev", None)
            # 缺省 prev → 视为势能不变（r=0），保证任何旧调用点都退回"无奖励"而非旧形式
            _gd_prev = _gd if _gd_prev is None else float(_gd_prev)
            r_goal_prox = _w_gp * (1.0 / (_gd + 0.5) - 1.0 / (_gd_prev + 0.5)) * s
        else:
            r_goal_prox = 0.0

        # ── 7. r_step: 微小时间税 — 抑制原地振荡 (2026-07-19 恢复) ──────
        r_step = self.step_penalty * s

        total = (
            r_cte
            + r_heading
            + r_obstacle
            + r_progress
            + r_reverse
            + r_steer
            + r_goal_prox
            + r_step
        )
        # 累计本回合稠密分（供终止时的"稠密累计量封顶"补偿）。
        # env 每步先调一次 done=False 的稠密 compute、终止时再调一次 done=True ——
        # env._step_count 在稠密调用之后才自增，所以**稠密调用的 steps_taken 与
        # 上一个稠密调用相同**，而终止调用带 steps_taken=n+1 ⇒ 恰好可用
        # "步数严格回退"识别新回合，不需要 env 侧显式 reset 钩子
        # （每步一个 env 进程、单线程使用，无并发问题）。
        _n_now = int(steps_taken) if steps_taken else 0
        if _n_now <= self._dense_steps:
            self._dense_sum = 0.0
            self._dense_steps = 0
        self._dense_sum += float(total)
        self._dense_steps = _n_now

        components = {
            # "path" 与 "progress" 为历史兼容别名，同值勿双计（test_logger 已相应处理）
            "path": float(r_progress),
            "heading": float(r_heading),
            "obstacle": float(r_obstacle),
            "cte": float(r_cte),
            "progress": float(r_progress),
            "reverse": float(r_reverse),
            "steer": float(r_steer),
            "goal_proximity": float(r_goal_prox),
            "alive": 0.0,
            "step_penalty": float(r_step),
            "low_speed": 0.0,  # 已移除
            "stop": 0.0,  # 已移除
            "curvature": 0.0,  # 已移除 (2026-07-18 v2)
            "collision": 0.0,
            "goal": 0.0,
            "goal_time": 0.0,
            "goal_heading": 0.0,
            "timeout": 0.0,
            "stuck": 0.0,
            "path_deviation": 0.0,
        }
        return float(total), components


# 所有奖励分量键名
_COMPONENT_KEYS = [
    "path",
    "heading",
    "obstacle",
    "cte",
    "progress",
    "reverse",
    "steer",
    "alive",
    "step_penalty",
    "low_speed",
    "goal_proximity",
    "stop",
    "curvature",
    "collision",
    "goal",
    "goal_heading",
    "goal_time",
    "timeout",
    "stuck",
    "path_deviation",
]


def _zero_comps(filled_key: str, total: float) -> dict[str, float]:
    """返回一个所有分量均为零的字典，仅 *filled_key* 除外。"""
    result = dict.fromkeys(_COMPONENT_KEYS, 0.0)
    result[filled_key] = total
    return result
