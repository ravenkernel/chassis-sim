"""Tests for simplified reward model (2026-07-18 — 4 components)."""

import numpy as np
import pytest

from chassis_sim.envs.reward import RewardCalculator


class TestTerminalRewards:
    """终止奖励：不变."""

    @pytest.fixture
    def calc(self):
        # 本类断言的是**公式与相对关系**（-100×stage_scale、goal_bonus+cos×10），
        # 与生产默认值无关，故显式给值：2026-10-23 默认值已按 config/reward.yaml
        # 对齐（goal_bonus 100→400、stuck_penalty -100→0），若不显式传入，
        # 这里测的就不再是公式而是"恰好等于某个标定值"。
        return RewardCalculator(goal_bonus=100.0, stuck_penalty=-100.0)

    def test_collision(self, calc):
        r, c = calc.compute(done=True, collision=True)
        # 2026-10-23 失败经济学：−100×stage_scale(0.2) − collision_surcharge(60) = −80
        assert r == pytest.approx(-80.0)
        assert c["collision"] == pytest.approx(-80.0)

    def test_goal_aligned(self, calc):
        r, _ = calc.compute(done=True, goal_reached=True, goal_heading_error=0.0)
        assert r == pytest.approx(110.0, rel=0.01)  # 100 + 1*1*10

    def test_goal_reversed(self, calc):
        r, _ = calc.compute(done=True, goal_reached=True, goal_heading_error=np.pi)
        assert r == pytest.approx(90.0, rel=0.01)  # 100 + 1*(-1)*10

    def test_goal_perpendicular(self, calc):
        r, _ = calc.compute(done=True, goal_reached=True, goal_heading_error=np.pi / 2)
        assert r == pytest.approx(100.0, rel=0.01)  # 100 + 1*0*10

    def test_stuck(self, calc):
        r, c = calc.compute(done=True, stuck=True)
        assert r == pytest.approx(-20.0)  # stage_scale(0)=0.2: -100 × 0.2
        assert c["stuck"] == pytest.approx(-20.0)

    def test_timeout_penalty_scales_with_completion(self, calc):
        """超时惩罚 = min(−timeout_min_penalty, −100×stage_scale×(1−arc_completion))。

        2026-10-23 失败经济学：timeout **也乘 stage_scale**（旧口径不乘 ⇒
        stage0 超时比撞墙还贵，策略学会用撞车结束回合），另加 −30 的**绝对值下限**
        （否则低阶段超时基项掉到 −20×(1−ac)，配上 +120 退款会让"悬停到超时"翻正）。
        """
        # stage 3 ⇒ s=1.0：完成度缩放原样可见（ac ≤ 0.7 时下限不介入）
        r, c = calc.compute(
            done=True, timeout=True, arc_completion=0.0, current_stage=3
        )
        assert r == pytest.approx(-100.0)
        assert c["timeout"] == pytest.approx(-100.0)
        r2, _ = calc.compute(
            done=True, timeout=True, arc_completion=0.3, current_stage=3
        )
        assert r2 == pytest.approx(-70.0)
        # 下限生效：ac ≥ 0.7 钉在 −30（旧口径会给 −10…−20）
        r3, _ = calc.compute(
            done=True, timeout=True, arc_completion=0.9, current_stage=3
        )
        assert r3 == pytest.approx(-30.0)
        r4, _ = calc.compute(
            done=True, timeout=True, arc_completion=1.0, current_stage=3
        )
        assert r4 == pytest.approx(-30.0)
        # stage 0 ⇒ s=0.2，−20×(1−ac) 永远够不到下限 ⇒ 全 ac 恒 −30
        r5, _ = calc.compute(done=True, timeout=True, arc_completion=0.0)
        assert r5 == pytest.approx(-30.0)
        r6, _ = calc.compute(done=True, timeout=True, arc_completion=0.9)
        assert r6 == pytest.approx(-30.0)

    def test_deviation(self, calc):
        r, c = calc.compute(done=True, path_deviation=True)
        assert r == pytest.approx(-20.0)  # stage_scale(0)=0.2
        assert c["path_deviation"] == pytest.approx(-20.0)


class TestPotentialProgress:
    """势能型 progress (2026-08-16 v5): w × (arc_pos_cur − arc_pos_prev)。"""

    @pytest.fixture
    def calc(self):
        # obstacle 权重归零以隔离 progress 语义：默认 d_min=10 时
        # -exp(-10/1.5)≈-0.0013 会泄漏进 total 断言 (abs=1e-6)
        return RewardCalculator(weights={"progress": 4.0, "obstacle": 0.0})

    def test_forward_positive(self, calc):
        """前进 0.2m: 4.0 × 0.2 = +0.8。"""
        _r, c = calc.compute(arc_progress=0.2)
        assert c["progress"] == pytest.approx(0.8, abs=0.01)
        assert c["path"] == pytest.approx(0.8, abs=0.01)

    def test_reverse_negative(self, calc):
        """后退 0.2m: 4.0 × (−0.2) = −0.8，不再截断为 0。"""
        r, _c = calc.compute(arc_progress=-0.2)
        assert r == pytest.approx(-0.8 - 0.02, abs=1e-6)  # progress + step_penalty

    def test_oscillation_net_zero(self, calc):
        """原地振荡 10 步: 前进+后退净 0。"""
        total = sum(calc.compute(arc_progress=0.1)[0] for _ in range(5)) + sum(
            calc.compute(arc_progress=-0.1)[0] for _ in range(5)
        )
        assert total == pytest.approx(10 * -0.02, abs=1e-6)  # 仅剩时间税


class TestQuadraticCTE:
    """二次型 CTE: -2.0 × CTE²."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator()

    def test_zero_cte_no_penalty(self, calc):
        """CTE=0 → 0."""
        _, c = calc.compute(cte=0.0, v=0.5, progress=0.5, v_max=1.2)
        assert c["cte"] == pytest.approx(0.0, abs=0.01)

    def test_small_cte_quadratic(self, calc):
        """CTE=0.15 → -2.0 × 0.0225 = -0.045 (near-zero gradient)."""
        _, c = calc.compute(cte=0.15, v=0.5, progress=0.5, v_max=1.2)
        assert c["cte"] == pytest.approx(-0.045, abs=0.01)

    def test_one_meter_quadratic(self, calc):
        """CTE=1.0 → -2.0 × 1.0 = -2.0."""
        _, c = calc.compute(cte=1.0, v=0.5, progress=0.5, v_max=1.2)
        assert c["cte"] == pytest.approx(-2.0, abs=0.01)

    def test_quadratic_scale(self, calc):
        """CTE doubled → penalty ×4 (quadratic)."""
        _, c1 = calc.compute(cte=0.5, v=0.5, progress=0.5, v_max=1.2)
        _, c2 = calc.compute(cte=1.0, v=0.5, progress=0.5, v_max=1.2)
        assert c2["cte"] == pytest.approx(c1["cte"] * 4.0, abs=0.01)

    def test_sign_symmetric(self, calc):
        """CTE sign irrelevant — square eliminates sign."""
        _, c_pos = calc.compute(cte=1.0, v=0.5, progress=0.5, v_max=1.2)
        _, c_neg = calc.compute(cte=-1.0, v=0.5, progress=0.5, v_max=1.2)
        assert c_pos["cte"] == pytest.approx(c_neg["cte"], abs=0.01)

    def test_four_meters_capped(self, calc):
        """CTE=4m → capped at cte_max=3.0 → -2.0 × 9.0 = -18.0."""
        _r, c = calc.compute(
            cte=4.0,
            v=1.2,
            progress=1.2,
            v_max=1.2,
            heading_error=0.0,
            heading_error_local=0.0,
        )
        assert c["cte"] == pytest.approx(-18.0, abs=0.5)

    def test_cte_at_cap(self, calc):
        """CTE=3.0 → -2.0 × 9.0 = -18.0 (at cap)."""
        _r, c = calc.compute(
            cte=3.0,
            v=1.2,
            progress=1.2,
            v_max=1.2,
            heading_error=0.0,
            heading_error_local=0.0,
        )
        assert c["cte"] == pytest.approx(-18.0, abs=0.5)

    def test_cte_beyond_cap(self, calc):
        """CTE=5.0 → still capped at -18.0 (same as CTE=3.0)."""
        _r3, c3 = calc.compute(
            cte=3.0,
            v=1.2,
            progress=1.2,
            v_max=1.2,
            heading_error=0.0,
            heading_error_local=0.0,
        )
        _r5, c5 = calc.compute(
            cte=5.0,
            v=1.2,
            progress=1.2,
            v_max=1.2,
            heading_error=0.0,
            heading_error_local=0.0,
        )
        assert c5["cte"] == pytest.approx(c3["cte"], abs=0.5)


class TestExponentialObstacle:
    """指数避障: -1.5 × exp(-d_min / 1.5)."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator(obstacle_range=1.5)

    def test_touching_max_penalty(self, calc):
        """d_min=0 → -1.0 × exp(0) = -1.0."""
        _, c = calc.compute(d_min=0.0, v=0.5, progress=0.5, v_max=1.2)
        assert c["obstacle"] == pytest.approx(-1.0, abs=0.01)

    def test_farther_is_better(self, calc):
        """Smooth monotonic decrease."""
        _, c_near = calc.compute(d_min=0.2, v=0.5, progress=0.5, v_max=1.2)
        _, c_far = calc.compute(d_min=1.5, v=0.5, progress=0.5, v_max=1.2)
        assert c_far["obstacle"] > c_near["obstacle"]

    def test_at_range_significant(self, calc):
        """d_min=1.5 → -1.0 × exp(-1) = -0.368 (not zero — smooth decay)."""
        _, c = calc.compute(d_min=1.5, v=0.5, progress=0.5, v_max=1.2)
        assert c["obstacle"] == pytest.approx(-0.368, abs=0.01)

    def test_far_approaches_zero(self, calc):
        """d_min=5.0 → -1.5 × exp(-5/1.5) = -0.053."""
        _, c = calc.compute(d_min=5.0, v=0.5, progress=0.5, v_max=1.2)
        assert c["obstacle"] > -0.06
        assert c["obstacle"] < 0.0

    def test_smooth_gradient(self, calc):
        """Gradient changes smoothly (no truncation jump)."""
        _, c0 = calc.compute(d_min=0.9, v=0.5, progress=0.5, v_max=1.2)
        _, c1 = calc.compute(d_min=1.0, v=0.5, progress=0.5, v_max=1.2)
        _, c2 = calc.compute(d_min=1.1, v=0.5, progress=0.5, v_max=1.2)
        diff_01 = abs(c0["obstacle"] - c1["obstacle"])
        diff_12 = abs(c1["obstacle"] - c2["obstacle"])
        assert abs(diff_01 - diff_12) < 0.05, "Gradient should be smooth"

    def test_near_goal_disables(self, calc):
        """near_goal=True → obstacle 归零."""
        _, c = calc.compute(d_min=0.0, near_goal=True, v=0.5, progress=0.5, v_max=1.2)
        assert c["obstacle"] == pytest.approx(0.0, abs=0.01)


class TestRemovedComponents:
    """已移除的分量均归零."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator()

    def test_heading_penalty_active(self, calc):
        """heading penalty now a real component — -0.3 × |heading_error|."""
        _, c = calc.compute(
            heading_error=0.0, heading_error_local=0.0, v=0.5, progress=0.5, v_max=1.2
        )
        assert c["heading"] == pytest.approx(0.0, abs=0.01)  # aligned → zero
        _, c = calc.compute(
            heading_error=np.pi / 2,
            heading_error_local=np.pi / 2,
            v=0.5,
            progress=0.5,
            v_max=1.2,
        )
        assert c["heading"] == pytest.approx(-0.471, abs=0.01)  # -0.3 × π/2

    def test_low_speed_removed(self, calc):
        """low_speed always zero."""
        _, c = calc.compute(v=0.0, progress=0.0, v_max=1.2, cte=0.0)
        assert c["low_speed"] == 0.0

    def test_stop_removed(self, calc):
        """stop always zero."""
        _, c = calc.compute(path_remaining=0.1, v=1.0, progress=1.0, v_max=1.2)
        assert c["stop"] == 0.0

    def test_legacy_keys_not_present(self, calc):
        """Old component keys absent."""
        _, c = calc.compute(cte=0.5, v=1.0, progress=1.0, v_max=1.2)
        for key in [
            "steer_effort",
            "smooth",
            "direction_change",
            "speed_tracking",
            "reverse_penalty",
            "r_path",
            "r_heading",
            "gate",
        ]:
            assert key not in c, f"{key} should be removed"

    def test_curvature_removed(self, calc):
        """curvature always zero."""
        _, c = calc.compute(v=0.5, progress=0.5, v_max=1.2, target_curve=0.3, delta=0.0)
        assert c["curvature"] == 0.0

    def test_step_penalty_active(self, calc):
        """step_penalty is -0.02 (restored 2026-07-19)."""
        _, c = calc.compute(v=1.0, progress=1.0, v_max=1.2)
        assert c["step_penalty"] == pytest.approx(-0.02, abs=0.01)


class TestStepPenalty:
    """微小时间税 — -0.02/步，抑制原地振荡."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator()

    def test_step_penalty_constant(self, calc):
        """Any state → step_penalty=-0.02."""
        _, c = calc.compute(v=0.0, progress=0.0, v_max=1.2)
        assert c["step_penalty"] == pytest.approx(-0.02, abs=0.01)

    def test_step_penalty_when_moving(self, calc):
        _, c = calc.compute(
            v=1.0, progress=1.0, v_max=1.2, heading_error=0.0, heading_error_local=0.0
        )
        assert c["step_penalty"] == pytest.approx(-0.02, abs=0.01)

    def test_step_penalty_zero_in_terminal(self, calc):
        _, c = calc.compute(done=True, goal_reached=True)
        assert c["step_penalty"] == pytest.approx(0.0, abs=0.01)


class TestReverseExploitDefense:
    """倒车 exploit 防御 (v5)：弧位置下降=负奖励，与车头朝向无关."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator(weights={"progress": 4.0, "obstacle": 0.0})

    def test_reverse_aligned_legitimate(self, calc):
        """前进段倒车：arc_progress=-0.5 → 4.0×(−0.5) = −2.0（正确惩罚倒车）。"""
        _, c = calc.compute(
            cte=0.0,
            heading_error=0.0,
            heading_error_local=0.0,
            v=-0.5,
            arc_progress=-0.5,
            v_max=1.2,
        )
        assert c["path"] == pytest.approx(-2.0, abs=0.01)  # 前进段倒车→负奖励

    def test_reverse_on_foldback_legitimate(self, calc):
        """折返段物理倒车=沿弧长前进：arc_progress=+0.5 → +2.0（正确奖励）。"""
        _, c = calc.compute(
            cte=0.0,
            heading_error=0.0,
            heading_error_local=np.pi,
            v=-0.5,
            arc_progress=0.5,
            v_max=1.2,
        )
        assert c["path"] == pytest.approx(2.0, abs=0.01)  # 折返段倒车→正奖励

    def test_forward_normal(self, calc):
        """正向行驶正常：arc_progress=1.2 → 4.0×1.2 = 4.8。"""
        _, c = calc.compute(
            cte=0.0,
            heading_error=0.0,
            heading_error_local=0.0,
            v=1.2,
            arc_progress=1.2,
            v_max=1.2,
        )
        assert c["path"] > 3.0


class TestArcCompletionIgnored:
    """arc_completion 不影响 collision/stuck/path_deviation/goal 惩罚；timeout 按完成度缩放 (2026-08-16 v5)。"""

    @pytest.fixture
    def calc(self):
        # stuck_penalty 显式给 -100：本类验证的是"arc_completion 不缩放 stuck"这条
        # 关系，而 2026-10-23 起默认值按 reward.yaml 对齐为 0（关闭 stuck），
        # 用默认值会让这条断言变成 0 == 0 的空检查。
        return RewardCalculator(stuck_penalty=-100.0)

    def test_collision_constant(self, calc):
        r0, _ = calc.compute(done=True, collision=True, arc_completion=0.0)
        r95, _ = calc.compute(done=True, collision=True, arc_completion=0.95)
        assert r0 == pytest.approx(-80.0)  # −100×stage_scale(0.2) − 60(附加罚)
        assert r0 == r95

    def test_timeout_scales_with_completion(self, calc):
        """超时惩罚随完成度缩放: −100×s×(1−ac)；s=1（stage 3）时 ac=0.3 ⇒ −70。

        用 stage 3 而非默认 stage 0：s=0.2 时 −20×(1−ac) 全被 −30 下限吃掉，
        缩放关系在该点不可见（"下限吃掉"本身由
        test_timeout_penalty_scales_with_completion 单独断言）。
        """
        r0, _ = calc.compute(
            done=True, timeout=True, arc_completion=0.0, current_stage=3
        )
        r30, _ = calc.compute(
            done=True, timeout=True, arc_completion=0.3, current_stage=3
        )
        assert r0 == pytest.approx(-100.0)
        assert r30 == pytest.approx(-70.0)
        assert r30 > r0

    def test_stuck_constant(self, calc):
        r0, _ = calc.compute(done=True, stuck=True, arc_completion=0.0)
        r60, _ = calc.compute(done=True, stuck=True, arc_completion=0.6)
        assert r0 == pytest.approx(-20.0)
        assert r0 == r60

    def test_path_deviation_constant(self, calc):
        r0, _ = calc.compute(done=True, path_deviation=True, arc_completion=0.0)
        r30, _ = calc.compute(done=True, path_deviation=True, arc_completion=0.3)
        assert r0 == pytest.approx(-20.0)
        assert r0 == r30

    def test_goal_unchanged(self, calc):
        r_full, _ = calc.compute(
            done=True,
            goal_reached=True,
            goal_heading_error=0.0,
            arc_completion=1.0,
            steps_taken=42,
            max_steps=500,
        )
        r_partial, _ = calc.compute(
            done=True,
            goal_reached=True,
            goal_heading_error=0.0,
            arc_completion=0.5,
            steps_taken=42,
            max_steps=500,
        )
        assert r_full == pytest.approx(r_partial)


class TestGoalTimeBonus:
    """Time-to-goal bonus."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator(goal_time_bonus=20.0)

    def test_fast_completion_max_bonus(self, calc):
        _r, c = calc.compute(
            done=True,
            goal_reached=True,
            goal_heading_error=0.0,
            steps_taken=42,
            max_steps=500,
        )
        expected_bonus = 20.0 * (1.0 - 42 / 500)
        assert c.get("goal_time", 0.0) == pytest.approx(expected_bonus, abs=0.01)

    def test_slow_completion_min_bonus(self, calc):
        _r, c = calc.compute(
            done=True,
            goal_reached=True,
            goal_heading_error=0.0,
            steps_taken=500,
            max_steps=500,
        )
        assert c.get("goal_time", 0.0) == pytest.approx(0.0, abs=0.01)

    def test_steps_exceeds_max_clamped(self, calc):
        _r, c = calc.compute(
            done=True,
            goal_reached=True,
            goal_heading_error=0.0,
            steps_taken=600,
            max_steps=500,
        )
        assert c.get("goal_time", 0.0) == pytest.approx(0.0, abs=0.01)

    def test_no_time_bonus_by_default(self):
        calc = RewardCalculator()
        _r, c = calc.compute(
            done=True,
            goal_reached=True,
            goal_heading_error=0.0,
            steps_taken=42,
            max_steps=500,
        )
        assert c.get("goal_time", 0.0) == 0.0

    def test_time_bonus_not_in_collision(self, calc):
        _r, c = calc.compute(done=True, collision=True, steps_taken=42, max_steps=500)
        assert c.get("goal_time", 0.0) == 0.0


class TestFastVsSlowStrategy:
    """快速行驶 per-step reward > 慢速."""

    def test_racing_beats_creeping(self):
        calc = RewardCalculator(goal_time_bonus=20.0)
        v_max = 1.2

        fast_r, _ = calc.compute(
            cte=0.0,
            heading_error=0.0,
            heading_error_local=0.0,
            d_min=10.0,
            v=1.2,
            progress=1.2,
            v_max=v_max,
        )
        slow_r, _ = calc.compute(
            cte=0.0,
            heading_error=0.0,
            heading_error_local=0.0,
            d_min=10.0,
            v=0.1,
            progress=0.1,
            v_max=v_max,
        )
        assert fast_r > slow_r * 3.0, (
            f"Fast per-step {fast_r:.2f} must be >3× slow {slow_r:.2f}"
        )


class TestBackwardCompat:
    """向后兼容：旧参数通过 **kwargs 忽略."""

    def test_old_params_accepted(self):
        calc = RewardCalculator(
            reverse_step_penalty=-0.2,
            path_d_max=2.0,
            curvature_scale=1.0,
            heading_alpha=3.0,
            speed_gate_threshold=0.10,
            direction_change_penalty=-1.0,
            progress_gate_mode="linear_vmax",
            progress_gate_floor=0.3,
            low_speed_weight=0.5,
            stop_threshold=0.2,
            obs_danger=0.3,
            cte_dead_zone=0.2,
            cross_track=0.5,
        )
        _r, c = calc.compute(cte=0.0, heading_error=0.0, v=1.0, progress=1.0, v_max=1.2)
        assert c["path"] > 0.0


class TestCurvatureRemoved:
    """曲率对齐奖励已移除 — curvature 恒零."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator()

    def test_curvature_always_zero(self, calc):
        """Any delta/target_curve → curvature=0."""
        _, c = calc.compute(v=0.5, progress=0.5, v_max=1.2, target_curve=0.3, delta=0.0)
        assert c["curvature"] == pytest.approx(0.0, abs=0.01)

    def test_curvature_zero_in_terminal(self, calc):
        _, c = calc.compute(done=True, goal_reached=True, goal_heading_error=0.0)
        assert c["curvature"] == pytest.approx(0.0, abs=0.01)
