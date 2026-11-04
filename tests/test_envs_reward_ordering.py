"""奖励排序与失败时刻补偿回归测试 (2026-10-05)。

覆盖三处结构性修复：
  ① steer 分量死代码：`kwargs.get('prev_delta')` 取不到命名参数 ⇒ 该项恒 0。
     修完必须满足"prev_delta 不同 ⇒ steer 项不同"。
  ② 失败时刻补偿（稠密累计量封顶）：失败代价不再随步数无界增长
     （"早死更便宜"消失），但**不得**出现"越差越好"（必须单调不减）。
     注：用户原提的"时间税补回"经实测方向相反（25/120 步失败差 7.5 → 9.8），
     已默认关闭（failure_time_compensation=0.0），保留为可调项。
  ③ 成功回合仍占优：goal ≈ +200 仍是最大正项，失败回合加补偿后仍远低于成功。
  ④ 端到端：env 的 timeout 终止分支必须把 steps_taken / max_steps 转发给
     RewardCalculator（此前不转发 ⇒ 补偿恒 0，修复在 timeout 路径上等于没做）。

注意：import numpy 前必须设 OMP_NUM_THREADS=1（否则 spawn 16 子进程时
OpenBLAS 拉线程池 → SIGKILL → ConnectionResetError [Errno 104]）。
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from chassis_sim.reward import RewardCalculator

WEIGHTS = {
    "cte_penalty": 3.0,
    "heading_penalty": 0.3,
    "heading_penalty_reverse": 0.5,
    "obstacle": 1.5,
    "progress": 4.0,
    "steer_penalty": 0.1,
    "collision": 1.0,
    "goal": 1.0,
    "goal_heading": 1.0,
    "stuck": 1.0,
    "timeout": 1.0,
    "path_deviation": 1.0,
    "goal_proximity": 2.0,
}


def _calc(**kw):
    # failure_dense_cap=0.12 显式传入：本文件验证的是**失败补偿机制本身**，
    # 而生产配置（config/reward.yaml）的取值历史上在 0.0 / 0.12 之间来回。
    # 不显式开启的话，下面 TestFailureTimeCompensation 的全部断言会退化成空检查。
    # failure_dense_refund_cap=0.0（=不限）同样显式钉住：2026-10-23 起
    # envs/reward.py 的签名默认 / RewardConfig 默认 / reward.yaml 都变成 120，
    # 而本文件要验证的是"退款 = |Σdense| − C 全额退还"这一机制（见
    # test_compensation_is_bounded_by_dense_defect 的等式断言）——带上限会把
    # 大额退款截断成 120，测的就不是机制本身了。上限的行为由
    # test_config_default_sync.py::TestFailureRefundBound 单独锁。
    params = {
        "weights": WEIGHTS,
        "obstacle_range": 1.5,
        "collision_penalty": -100.0,
        "goal_bonus": 200.0,
        "goal_heading_scale": 10.0,
        "stuck_penalty": 0.0,
        "timeout_penalty": -100.0,
        "path_deviation_penalty": -100.0,
        "goal_time_bonus": 0.0,
        "step_penalty": -0.02,
        "delta_max": 0.526,
        "failure_dense_cap": 0.12,
        "failure_dense_refund_cap": 0.0,
    }
    params.update(kw)
    return RewardCalculator(**params)


class TestSteerTermAlive:
    """① steer 分量必须真的随 prev_delta 变化（修复死代码）。"""

    def test_prev_delta_changes_steer_component(self):
        c = _calc()
        _, c0 = c.compute(d_min=100.0, prev_delta=0.0)
        _, c1 = c.compute(d_min=100.0, prev_delta=0.526)
        assert c0["steer"] == 0.0
        # -0.1 × 0.526 = -0.0526
        assert c1["steer"] == pytest.approx(-0.0526, abs=1e-6)
        assert c1["steer"] != c0["steer"], "prev_delta 不同必须给出不同的 steer 项"

    def test_total_differs_with_prev_delta(self):
        """旧 bug 的指纹：两个 prev_delta 的 total 逐位相同。"""
        c = _calc()
        r0, _ = c.compute(d_min=100.0, prev_delta=0.0)
        r1, _ = c.compute(d_min=100.0, prev_delta=0.526)
        assert r1 == pytest.approx(r0 - 0.0526, abs=1e-6)
        assert r1 != r0

    def test_steer_scales_with_weight_and_dense_scale(self):
        c = _calc(weights=dict(WEIGHTS, steer_penalty=0.4))
        _, comp = c.compute(d_min=100.0, prev_delta=0.5)
        assert comp["steer"] == pytest.approx(-0.2, abs=1e-9)
        c0 = _calc(weights=dict(WEIGHTS, steer_penalty=0.0))
        _, comp0 = c0.compute(d_min=100.0, prev_delta=0.5)
        assert comp0["steer"] == 0.0

    def test_steer_is_penalty_for_both_signs(self):
        c = _calc()
        _, cp = c.compute(d_min=100.0, prev_delta=0.3)
        _, cn = c.compute(d_min=100.0, prev_delta=-0.3)
        assert cp["steer"] == pytest.approx(cn["steer"], abs=1e-12)
        assert cp["steer"] < 0.0


class TestFailureTimeCompensation:
    """② 终止惩罚不再随步数单调变化；且不出现"越差越好"。"""

    def _dense_run(self, calc, n, per_step=None):
        """跑 n 步稠密奖励，返回 (Σ稠密, 末步 steps_taken)。"""
        total = 0.0
        for i in range(n):
            kw = {"d_min": 100.0, "arc_progress": 0.05, "steps_taken": i}
            if per_step:
                kw.update(per_step)
            r, _ = calc.compute(**kw)
            total += r
        return total

    def test_collision_cost_similar_for_early_and_late_failure(self):
        """同一条轨迹上 25 步失败 ≈ 116 步失败（旧差 382 量级）。

        稠密速率固定为 -4.18/步（用户实测 stage4 的数字）：靠 cte 制造，
        cte 使 -3.0×min(CTE²,9) = -4.18 ⇒ CTE²=1.393。
        """
        cte = float(np.sqrt(1.393))
        early = _calc(failure_dense_cap=0.0, failure_time_compensation=0.0)
        s25 = self._dense_run(early, 25, {"cte": cte, "steps_taken": 0})
        s116 = 0.0
        c = _calc(failure_dense_cap=0.0, failure_time_compensation=0.0)
        for i in range(116):
            r, _ = c.compute(d_min=100.0, cte=cte, steps_taken=i)
            s116 += r
        r25_old = (
            early.compute(
                done=True,
                collision=True,
                steps_taken=25,
                max_steps=500,
                current_stage=4,
            )[0]
            + s25
        )
        c2 = _calc(failure_dense_cap=0.0, failure_time_compensation=0.0)
        s116b = 0.0
        for i in range(116):
            r, _ = c2.compute(d_min=100.0, cte=cte, steps_taken=i)
            s116b += r
        r116_old = (
            c2.compute(
                done=True,
                collision=True,
                steps_taken=116,
                max_steps=500,
                current_stage=4,
            )[0]
            + s116b
        )
        old_gap = abs(r25_old - r116_old)
        assert old_gap > 300.0, f"前提校验：旧行为应当有很大差（实测 {old_gap:.0f}）"

        # 新奖励（线上默认：failure_dense_cap=0.12 ⇒ C=0.12×500=60）
        def ep(n):
            cc = _calc()
            s = 0.0
            for i in range(n):
                r, _ = cc.compute(d_min=100.0, cte=cte, steps_taken=i)
                s += r
            term, _ = cc.compute(
                done=True, collision=True, steps_taken=n, max_steps=500, current_stage=4
            )
            return s + term

        new_gap = abs(ep(25) - ep(116))
        assert new_gap < 30.0, (
            f"新行为差应当收敛（实测 {new_gap:.1f}，旧 {old_gap:.0f}）"
        )
        assert new_gap < old_gap / 10.0

    def test_compensation_is_monotone_in_dense_defect(self):
        """补偿不得造成"越差越好"：稠密缺陷越大，失败总回报越低（或持平）。"""
        cte = float(np.sqrt(1.393))
        prev = None
        for n in (20, 40, 80, 160, 320):
            cc = _calc()
            s = 0.0
            for i in range(n):
                r, _ = cc.compute(d_min=100.0, cte=cte, steps_taken=i)
                s += r
            tot = (
                s
                + cc.compute(
                    done=True,
                    collision=True,
                    steps_taken=n,
                    max_steps=500,
                    current_stage=4,
                )[0]
            )
            if prev is not None:
                assert tot <= prev + 1e-6, (
                    f"步数 {n} 的失败回报 {tot:.1f} 高于更短失败 {prev:.1f}——"
                    "补偿造成非单调（越差越好）"
                )
            prev = tot

    def test_dense_cap_zero_disables_compensation(self):
        """cap=0 且 ftc=0 ⇒ 完全退回旧行为（可 A/B 回滚）。"""
        cte = float(np.sqrt(1.393))
        for n in (25, 116):
            cc = _calc(failure_dense_cap=0.0, failure_time_compensation=0.0)
            s = 0.0
            for i in range(n):
                r, _ = cc.compute(d_min=100.0, cte=cte, steps_taken=i)
                s += r
            term, comps = cc.compute(
                done=True, collision=True, steps_taken=n, max_steps=500, current_stage=4
            )
            # stage4 ⇒ stage_scale=1.0；2026-10-23 起碰撞还要再减
            # collision_surcharge(60) ⇒ −160（旧口径 −100）
            assert term == pytest.approx(-160.0, abs=1e-9)
            assert comps["collision"] == pytest.approx(-160.0, abs=1e-9)

    def test_collision_penalty_bounded_by_cap(self):
        """补偿有界：|失败回报| 不再随步数无界增长（cap 之后持平）。"""
        cte = float(np.sqrt(1.393))
        tots = []
        for n in (120, 240, 480):
            cc = _calc()
            s = 0.0
            for i in range(n):
                r, _ = cc.compute(d_min=100.0, cte=cte, steps_taken=i)
                s += r
            tots.append(
                s
                + cc.compute(
                    done=True,
                    collision=True,
                    steps_taken=n,
                    max_steps=500,
                    current_stage=4,
                )[0]
            )
        # C = 0.12×500 = 60 ⇒ 长失败代价被钉在 -(100+60+10) 量级附近
        assert max(tots) - min(tots) < 40.0, f"长失败仍在无界恶化: {tots}"

    def test_no_compensation_on_success(self):
        """成功分支不补偿：goal 奖励与步数无关（时间奖金已关）。"""
        c = _calc()
        r_fast, comps = c.compute(
            done=True,
            goal_reached=True,
            steps_taken=30,
            max_steps=500,
            goal_heading_error=0.0,
        )
        c2 = _calc()
        r_slow, _ = c2.compute(
            done=True,
            goal_reached=True,
            steps_taken=400,
            max_steps=500,
            goal_heading_error=0.0,
        )
        assert r_fast == pytest.approx(r_slow, abs=1e-9)
        assert comps["goal_time"] == 0.0
        assert r_fast == pytest.approx(210.0, abs=1e-6)  # 200 + 10×cos(0)

    def test_timeout_branch_also_compensated(self):
        """timeout 分支补偿也生效（需要 steps_taken 被转发）。

        稠密缺陷 487 ≫ C=60 ⇒ 终止项 = 基项(-48) + 封顶退回(+427)
        ⇒ 明显**高于**未补偿的基项。这正是"失败代价不随发生时刻变化"
        需要的量级（未补偿时 25 步失败 -85 / 116 步失败 -55，补偿后两者
        都收敛到 ≈ -(100+C) 附近）。
        """
        cte = float(np.sqrt(1.393))
        cc = _calc()
        for i in range(116):
            cc.compute(d_min=100.0, cte=cte, steps_taken=i)
        r_off = -100.0 * (1.0 - 0.52)
        term, _ = cc.compute(
            done=True,
            timeout=True,
            arc_completion=0.52,
            steps_taken=116,
            max_steps=500,
            current_stage=4,
        )
        assert term > r_off + 80.0, f"timeout 补偿未生效: {term:.3f} vs {r_off:.3f}"
        # 补偿有界：C = 0.12×500 = 60 ⇒ 终止调整 = |Σdense| − 60 量级
        assert term < 700.0

    def test_compensation_is_bounded_by_dense_defect(self):
        """补偿有界：0 ≤ comp ≤ |Σ稠密|（封顶只退"超额"部分，且随缺陷单调）。"""
        cte = float(np.sqrt(1.393))
        cc = _calc()
        dsum = 0.0
        for i in range(300):
            r, _ = cc.compute(d_min=100.0, cte=cte, steps_taken=i)
            dsum += r
        term, _ = cc.compute(
            done=True,
            timeout=True,
            arc_completion=0.3,
            steps_taken=300,
            max_steps=500,
            current_stage=4,
        )
        base = -100.0 * (1.0 - 0.3)
        comp = term - base
        assert 0.0 <= comp <= abs(dsum) + 1e-6
        _cap = 0.12 * 500.0
        assert comp == pytest.approx(-dsum - _cap, rel=1e-6)

    def test_time_tax_compensation_off_by_default(self):
        """时间税补回默认关闭（实测方向与目标相反：25/120 步差 7.5 → 9.8）。"""
        c = RewardCalculator()
        assert c.failure_time_compensation == pytest.approx(0.0)
        # 打开它必须真的改变终止项（可调、可回滚）
        cte = float(np.sqrt(1.393))
        cc = _calc(failure_time_compensation=1.0)
        for i in range(100):
            cc.compute(d_min=100.0, cte=cte, steps_taken=i)
        term_on, _ = cc.compute(
            done=True,
            timeout=True,
            arc_completion=0.5,
            steps_taken=100,
            max_steps=500,
            current_stage=4,
        )
        cc2 = _calc(failure_time_compensation=0.0)
        for i in range(100):
            cc2.compute(d_min=100.0, cte=cte, steps_taken=i)
        term_off, _ = cc2.compute(
            done=True,
            timeout=True,
            arc_completion=0.5,
            steps_taken=100,
            max_steps=500,
            current_stage=4,
        )
        assert term_on == pytest.approx(term_off - 0.02 * 100, rel=1e-9)


class TestSuccessStillDominates:
    """③ 成功回合的相对优势必须保持（goal ≈ +200 是最大正项）。"""

    def test_goal_beats_best_possible_failure(self):
        c = _calc()
        goal, _ = c.compute(
            done=True,
            goal_reached=True,
            steps_taken=400,
            max_steps=500,
            goal_heading_error=0.0,
        )
        # 最"好心"的失败：0 步稠密缺陷 + 全额补偿
        c2 = _calc()
        coll, _ = c2.compute(
            done=True, collision=True, steps_taken=1, max_steps=500, current_stage=2
        )
        assert goal > coll + 250.0, f"成功 {goal} 必须显著优于失败 {coll}"

    def test_goal_bonus_unchanged_by_config_defaults(self):
        c = _calc()
        r, comps = c.compute(done=True, goal_reached=True, goal_heading_error=0.0)
        assert comps["goal"] == pytest.approx(200.0, abs=1e-9)
        assert r == pytest.approx(210.0, abs=1e-6)

    def test_ordering_flip_correct_failure_beats_shortcut(self):
        """正确沿路径但失败 > 抄近道失败（同一步数、同一失败类型）。"""
        # 正确：沿路径 ⇒ CTE≈0、有进度势能
        good = _calc()
        s_good = 0.0
        for i in range(120):
            r, _ = good.compute(d_min=100.0, arc_progress=0.05, cte=0.0, steps_taken=i)
            s_good += r
        good_total = (
            s_good
            + good.compute(
                done=True,
                collision=True,
                steps_taken=120,
                max_steps=500,
                current_stage=4,
            )[0]
        )

        # 抄近道：CTE 大 + 无沿路径进度
        bad = _calc()
        s_bad = 0.0
        for i in range(120):
            r, _ = bad.compute(d_min=100.0, arc_progress=0.0, cte=1.5, steps_taken=i)
            s_bad += r
        bad_total = (
            s_bad
            + bad.compute(
                done=True,
                collision=True,
                steps_taken=120,
                max_steps=500,
                current_stage=4,
            )[0]
        )

        assert good_total > bad_total, (
            f"正确沿路径失败 {good_total:.1f} 必须优于抄近道失败 {bad_total:.1f}"
        )


class TestDefaultsAndPackagedYaml:
    """④ 默认值三方一致 + 包内 reward.yaml（抽包后的端到端等价物）。

    与主仓同名类的差异：主仓这里还有 test_timeout_terminal_call_carries_steps，
    它读 envs/ackermann_env.py 的源码断言 timeout/stuck/path_deviation 终止分支
    转发了 steps_taken / max_steps —— 那是 **env 主体**的职责，本包只移植数据层
    （不搬 ackermann_env.py），故该断言不随包发布（见回报"未做项"）。
    补偿机制本身由上文的 TestFailureTimeCompensation 全量覆盖。
    """

    def test_reward_calculator_defaults_are_active(self):
        c = RewardCalculator()
        assert c.failure_time_compensation == pytest.approx(0.0)  # 默认关（见类注释）
        # 该值随 2026-10-23 诊断调整：failure_dense_cap 0.0 → 0.12、
        # failure_dense_refund_cap 0.0 → 120.0（run100 实测碰撞 −57.2/37 步 vs
        # 超时 −302.1/300 步 ⇒ 早死便宜 5.3 倍）。三方一致（reward.yaml /
        # RewardConfig / RewardCalculator 签名）由
        # tests/test_config_default_sync.py::test_three_way_defaults_agree 锁住。
        assert c.failure_dense_cap == pytest.approx(0.12)  # 速率：C = 0.12×max_steps
        assert c.failure_dense_refund_cap == pytest.approx(120.0)  # 单次退款上限
        assert c.step_penalty == pytest.approx(-0.02)

    def test_packaged_reward_yaml_has_goal_time_bonus_zero(self):
        """读**包内**（随 wheel 发布的 package data）reward.yaml，而不是 CWD 下的。"""
        import importlib.resources

        import yaml

        res = importlib.resources.files("chassis_sim.config").joinpath("reward.yaml")
        assert res.is_file(), "reward.yaml 必须作为 package data 随 wheel 发布"
        cfg = yaml.safe_load(res.read_text(encoding="utf-8"))
        assert float(cfg["reward_params"]["goal_time_bonus"]) == 0.0, (
            "时间红利必须关闭（否则抄近道净赚 ≈ +47/回合）"
        )


class TestFailureRefundBound:
    """残余风险锁（主仓 2026-10-23，移植自 tests/test_config_default_sync.py:419-487）：
    退款上限是「终止奖励不得为大正数」的**唯一**保证。

    失败补偿是终止步的一次性退款 +min(|Σdense| − C, refund_cap)。没有上限时
    reward.py 的注释记录过 +3562 的尖峰（= goal_bonus 的 8.9×），它把「长失败回路
    的前一步」变成 MDP 里价值最高的状态，并让 4.2% 的 target 撞上 +150 截断。
    这里遍历 stage × |Σdense| × 终止类型，断言上界恒成立：
        collision（comp_scale = stage_scale）:
            ≤ −100×stage_scale − collision_surcharge + 120×stage_scale
        timeout  （comp_scale = 1.0，不乘 stage_scale）:
            ≤ min(−timeout_min_penalty, −100×stage_scale×(1−ac)) + 120
    ⚠ 退款上限 **不随 stage_scale 缩放**（reward.py 的 min 作用在已乘 comp_scale 的
    _refund 上）⇒ stage0 的长失败碰撞上界是 −20−60+120 = +40，而不是 −80+24。
    全局最大值是 **+90**（timeout 的 ac≥0.7 档：基项被 −timeout_min_penalty(30)
    钉在 −30，退满 120 ⇒ +90）；碰撞侧最高只到 +40（stage0）。

    移植差异：主仓经 training.env_factory.make_env(..., reward_cfg=None) 取默认
    calculator；本包没有 env 工厂，直接 RewardCalculator()。两者等价的前提是
    三方默认值一致，那正是本文件 TestDefaultsAndPackagedYaml 钉住的内容。
    """

    def test_terminal_reward_never_exceeds_base_plus_refund_cap(self):
        import copy

        calc = RewardCalculator()
        assert calc.failure_dense_refund_cap > 0.0, "上限必须开着，否则本节无意义"
        cap = float(calc.failure_dense_refund_cap)
        C = float(calc.failure_dense_cap) * 500.0  # = 60 @max_steps=500
        seen = []
        for stage in (0, 1, 2, 3, 4, 5):
            stage_scale = min(max(stage / 3.0, 0.2), 1.0)
            for mag in (0.0, 60.0, 180.0, 500.0, 5000.0):
                for kind, ac in (
                    ("collision", 0.0),
                    ("timeout", 0.0),
                    ("timeout", 0.05),
                    ("timeout", 0.5),
                    ("timeout", 0.9),
                ):
                    c = copy.deepcopy(calc)
                    c._dense_sum, c._dense_steps = -float(mag), 0  # 直接给 Σdense
                    r, _ = c.compute(
                        done=True,
                        collision=(kind == "collision"),
                        timeout=(kind == "timeout"),
                        steps_taken=301,
                        max_steps=500,
                        current_stage=stage,
                        arc_completion=ac,
                    )
                    if kind == "collision":
                        base = -100.0 * stage_scale - float(calc.collision_surcharge)
                        scale = stage_scale
                    else:
                        base = min(
                            -float(calc.timeout_min_penalty),
                            -100.0 * stage_scale * (1.0 - ac),
                        )
                        scale = 1.0
                    seen.append(float(r))
                    # 退款 = min(scale × max(0, |Σdense| − C), cap)：注意 cap 是在
                    # **乘完 comp_scale 之后**才截断的（reward.py 的 min 作用在
                    # _refund 上），所以 stage0 的长失败碰撞能到 −20−60+120 = +40。
                    refund = min(scale * max(0.0, mag - C), cap)
                    expected = base + refund
                    assert r == pytest.approx(expected, abs=1e-6), (
                        f"终止奖励 {r:+.2f} ≠ 基项 {base:+.2f} + 退款 {refund:+.2f}"
                        f"（stage={stage} |Σdense|={mag:.0f} {kind} ac={ac}）"
                    )
                    assert r <= base + cap + 1e-9, (
                        f"终止奖励 {r:+.2f} 超过绝对上界 {base + cap:+.2f} —— 退款上限失效"
                    )
        assert max(seen) == pytest.approx(90.0, abs=1e-6), (
            f"退款上界的实测最大值变了（现 {max(seen):+.2f}）；"
            "若有意改 refund_cap / timeout_min_penalty，请同步这里与 reward.yaml 的注释"
        )

    def test_refund_hits_cap_when_dense_defect_far_exceeds_cap(self):
        """refund_cap 边界（一）：|Σdense| ≫ C 时退款恰好等于 refund_cap。

        等价于主仓 tests/test_config_default_sync.py:400-414（400 步 × cte 0.5
        ⇒ 稠密 ≈ −1.5/步 ⇒ |Σdense| ≈ 602 > C=60 ⇒ 退款被 120 截断）。
        """
        calc = RewardCalculator()
        for i in range(400):
            calc.compute(
                cte=0.5,
                heading_error=0.0,
                d_min=10.0,
                progress=0.0,
                prev_delta=0.0,
                steps_taken=i,
                max_steps=500,
                current_stage=4,
            )
        total, comps = calc.compute(
            done=True, collision=True, steps_taken=401, max_steps=500, current_stage=4
        )
        base = calc.collision_penalty - float(calc.collision_surcharge)  # = −160
        assert total == pytest.approx(base + calc.failure_dense_refund_cap, abs=1e-6), (
            f"stage4 碰撞终止奖励应为 基项 {base:+.0f} + 退款上限 120 = {base + 120:+.0f}，"
            f"实得 {total:+.2f}"
        )
        assert comps["collision"] == pytest.approx(total)
        assert total <= base + calc.failure_dense_refund_cap + 1e-9, (
            "退款超过了 failure_dense_refund_cap —— 上限失效（历史 +3562 尖峰会回来）"
        )

    def test_no_refund_when_dense_defect_below_cap(self):
        """cap 边界（二）：|Σdense| ≤ C ⇒ 退款为 0（补偿只在超额时发生）。

        10 步 cte=0.1 ⇒ 每步 −6×0.01 − 0.02 ≈ −0.08 ⇒ |Σdense| ≈ 0.8 ≪ C=60。
        """
        calc = RewardCalculator()
        for i in range(10):
            calc.compute(
                cte=0.1, d_min=10.0, steps_taken=i, max_steps=500, current_stage=4
            )
        total, _ = calc.compute(
            done=True,
            timeout=True,
            arc_completion=0.5,
            steps_taken=10,
            max_steps=500,
            current_stage=4,
        )
        # timeout 基项 = min(−30, −100×1.0×(1−0.5)) = −50，无退款。
        assert total == pytest.approx(-50.0, abs=1e-6), (
            f"未超额却发生退款：终止奖励 {total:+.3f} ≠ −50"
        )

    def test_refund_cap_value_sets_the_hard_ceiling(self):
        """refund_cap 边界（三）：同一条轨迹上 cap=120 与 cap=50 的差额 = 70。"""
        cte = float(np.sqrt(1.393))

        def ep(refund_cap):
            c = _calc(failure_dense_refund_cap=refund_cap)
            for i in range(300):
                c.compute(d_min=100.0, cte=cte, steps_taken=i)
            term, _ = c.compute(
                done=True,
                timeout=True,
                arc_completion=0.3,
                steps_taken=300,
                max_steps=500,
                current_stage=4,
            )
            return term

        wide, narrow = ep(120.0), ep(50.0)
        base = -100.0 * (1.0 - 0.3)
        assert wide - base == pytest.approx(120.0, abs=1e-6)
        assert narrow - base == pytest.approx(50.0, abs=1e-6)
        assert wide - narrow == pytest.approx(70.0, abs=1e-6)
