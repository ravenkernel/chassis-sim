# ruff: noqa: F841, I001
# fmt: off
"""Reeds-Shepp curve implementation for Ackermann vehicle path generation.

Embedded pure-Python implementation. Generates optimal paths respecting
a minimum turning radius with natural forward/backward segments.
"""
import math
import numpy as np
from chassis_sim.paths import angle_wrap


def reeds_shepp_path(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    radius: float,
    num_points: int = 20,
) -> tuple[np.ndarray, list[dict]]:
    """Compute RS path from start to goal pose.

    Args:
        start: (x, y, theta) start pose in world frame.
        goal: (x, y, theta) goal pose in world frame.
        radius: minimum turning radius in meters.
        num_points: number of waypoints to sample.

    Returns:
        path_waypoints: (num_points, 2) sampled points in world coords.
        segments: list of segment dicts with keys:
            type: 'forward' | 'backward'
            length: float (meters)
            curvature: float (-1/r, 0, or 1/r)
            direction: 'left' | 'straight' | 'right'
    """
    sx, sy, st = float(start[0]), float(start[1]), float(start[2])
    gx, gy, gt = float(goal[0]), float(goal[1]), float(goal[2])
    r = float(radius)

    cos_st = np.cos(st)
    sin_st = np.sin(st)
    dx = gx - sx
    dy = gy - sy
    gx_rel = dx * cos_st + dy * sin_st
    gy_rel = -dx * sin_st + dy * cos_st
    gt_rel = angle_wrap(gt - st)

    segments_rel = _solve_rs(gx_rel, gy_rel, gt_rel, r)
    if segments_rel is None:
        return None, []

    segments = []
    for seg in segments_rel:
        direction, curvature, length = seg
        if curvature > 0:
            direction_side = 'left'
        elif curvature < 0:
            direction_side = 'right'
        else:
            direction_side = 'straight'
        segments.append({
            'type': direction,
            'length': length,
            'curvature': curvature,
            'direction': direction_side,
        })

    path = _sample_segments(sx, sy, st, segments_rel, num_points)
    return path, segments


def reeds_shepp_length(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    radius: float,
) -> float:
    """Analytic RS path length (no path generation, no collision check)."""
    sx, sy, st = float(start[0]), float(start[1]), float(start[2])
    gx, gy, gt = float(goal[0]), float(goal[1]), float(goal[2])
    r = float(radius)
    cos_st, sin_st = math.cos(st), math.sin(st)
    dx, dy = gx - sx, gy - sy
    gx_rel = dx * cos_st + dy * sin_st
    gy_rel = -dx * sin_st + dy * cos_st
    gt_rel = angle_wrap(gt - st)
    segs = _solve_rs(gx_rel, gy_rel, gt_rel, r)
    if segs is None:
        return float('inf')
    return float(sum(abs(s[2]) for s in segs))


def _solve_rs(x, y, phi, r):
    best_length = float('inf')
    best_segs = None
    for mode_fn in _RS_MODES:
        segs = mode_fn(x, y, phi, r)
        if segs is not None:
            length = sum(abs(s[2]) for s in segs)
            if length < best_length:
                best_length = length
                best_segs = segs
    if best_segs is None:
        # No RS mode works — return None and let caller handle fallback
        return None
    return best_segs


def _polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


# CSC modes (curve-straight-curve)

def _lsl(x, y, phi, r):
    u, t = _polar(x - r * np.sin(phi), y - r * (1 - np.cos(phi)))
    if t < -1e-12:
        return None
    v = angle_wrap(phi - t)
    if v < 0.0:
        return None
    return [
        ('forward', 1/r, t * r),
        ('forward', 0.0, u),
        ('forward', 1/r, v * r),
    ]

def _lsr(x, y, phi, r):
    u, t = _polar(x + r * np.sin(phi), y - r * (1 + np.cos(phi)))
    u1 = u * u
    t1 = 4 * r * r
    if u1 < t1:
        return None
    tau = math.sqrt(u1 - t1)
    alpha = np.arcsin(2 * r / u)
    t = t + alpha  # corrected: t must include the asin contribution
    v = angle_wrap(t - phi)
    if t < -1e-12 or v < -1e-12:
        return None
    return [
        ('forward', 1/r, t * r),
        ('forward', 0.0, tau),
        ('forward', -1/r, v * r),
    ]

def _rsl(x, y, phi, r):
    u, t = _polar(x - r * np.sin(phi), y + r * (1 + np.cos(phi)))
    u1 = u * u
    t1 = 4 * r * r
    if u1 < t1:
        return None
    tau = math.sqrt(u1 - t1)
    alpha = np.arcsin(2 * r / u)
    t = t + alpha  # corrected: t must include the asin contribution
    v = angle_wrap(t - phi)
    if t < -1e-12 or v < -1e-12:
        return None
    return [
        ('forward', -1/r, t * r),
        ('forward', 0.0, tau),
        ('forward', 1/r, v * r),
    ]

def _rsr(x, y, phi, r):
    u, t = _polar(x + r * np.sin(phi), y + r * (1 - np.cos(phi)))
    if t < -1e-12:
        return None
    v = angle_wrap(phi - t)
    if v < -1e-12:
        return None
    return [
        ('forward', -1/r, t * r),
        ('forward', 0.0, u),
        ('forward', -1/r, v * r),
    ]

# CSC mixed-direction modes: C⁺S⁺C⁻ (overshoot & back in) and C⁻S⁺C⁺ (back out & drive).
# Same geometric path as the all-forward CSC, only the heading constraint changes
# because backward segments flip the sign of dθ contribution.

def _lsl_ffb(x, y, phi, r):
    """L⁺S⁺L⁻: left forward, straight, left backward."""
    u, t = _polar(x - r * np.sin(phi), y - r * (1 - np.cos(phi)))
    if t < -1e-12:
        return None
    v = angle_wrap(t - phi)  # dθ: +t + 0 - v = phi → v = t - φ
    if v < -1e-12:
        return None
    return [
        ('forward', 1/r, t * r),
        ('forward', 0.0, u),
        ('backward', 1/r, v * r),
    ]

def _lsr_ffb(x, y, phi, r):
    """L⁺S⁺R⁻: left forward, straight, right backward."""
    u, t = _polar(x + r * np.sin(phi), y - r * (1 + np.cos(phi)))
    u1 = u * u
    t1 = 4 * r * r
    if u1 < t1:
        return None
    tau = math.sqrt(u1 - t1)
    alpha = np.arcsin(2 * r / u)
    t = t + alpha
    v = angle_wrap(phi - t)  # dθ: +t + 0 + v = φ → v = φ - t
    if t < -1e-12 or v < -1e-12:
        return None
    return [
        ('forward', 1/r, t * r),
        ('forward', 0.0, tau),
        ('backward', -1/r, v * r),
    ]

def _rsl_ffb(x, y, phi, r):
    """R⁺S⁺L⁻: right forward, straight, left backward."""
    u, t = _polar(x - r * np.sin(phi), y + r * (1 + np.cos(phi)))
    u1 = u * u
    t1 = 4 * r * r
    if u1 < t1:
        return None
    tau = math.sqrt(u1 - t1)
    alpha = np.arcsin(2 * r / u)
    t = t + alpha
    v = angle_wrap(-phi - t)  # dθ: -t + 0 - v = φ → v = -φ - t
    if t < -1e-12 or v < -1e-12:
        return None
    return [
        ('forward', -1/r, t * r),
        ('forward', 0.0, tau),
        ('backward', 1/r, v * r),
    ]

def _rsr_ffb(x, y, phi, r):
    """R⁺S⁺R⁻: right forward, straight, right backward."""
    u, t = _polar(x + r * np.sin(phi), y + r * (1 - np.cos(phi)))
    if t < -1e-12:
        return None
    v = angle_wrap(phi + t)  # dθ: -t + 0 + v = φ → v = φ + t
    if v < -1e-12:
        return None
    return [
        ('forward', -1/r, t * r),
        ('forward', 0.0, u),
        ('backward', -1/r, v * r),
    ]

def _lsl_bff(x, y, phi, r):
    """L⁻S⁺L⁺: left backward, straight, left forward."""
    u, t = _polar(x - r * np.sin(phi), y - r * (1 - np.cos(phi)))
    if t < -1e-12:
        return None
    v = angle_wrap(phi + t)  # dθ: -t + 0 + v = φ → v = φ + t
    if v < -1e-12:
        return None
    return [
        ('backward', 1/r, t * r),
        ('forward', 0.0, u),
        ('forward', 1/r, v * r),
    ]

def _lsr_bff(x, y, phi, r):
    """L⁻S⁺R⁺: left backward, straight, right forward."""
    u, t = _polar(x + r * np.sin(phi), y - r * (1 + np.cos(phi)))
    u1 = u * u
    t1 = 4 * r * r
    if u1 < t1:
        return None
    tau = math.sqrt(u1 - t1)
    alpha = np.arcsin(2 * r / u)
    t = t + alpha
    v = angle_wrap(-phi - t)  # dθ: -t + 0 - v = φ → v = -φ - t
    if t < -1e-12 or v < -1e-12:
        return None
    return [
        ('backward', 1/r, t * r),
        ('forward', 0.0, tau),
        ('forward', -1/r, v * r),
    ]

def _rsl_bff(x, y, phi, r):
    """R⁻S⁺L⁺: right backward, straight, left forward."""
    u, t = _polar(x - r * np.sin(phi), y + r * (1 + np.cos(phi)))
    u1 = u * u
    t1 = 4 * r * r
    if u1 < t1:
        return None
    tau = math.sqrt(u1 - t1)
    alpha = np.arcsin(2 * r / u)
    t = t + alpha
    v = angle_wrap(phi - t)  # dθ: +t + 0 + v = φ → v = φ - t
    if t < -1e-12 or v < -1e-12:
        return None
    return [
        ('backward', -1/r, t * r),
        ('forward', 0.0, tau),
        ('forward', 1/r, v * r),
    ]

def _rsr_bff(x, y, phi, r):
    """R⁻S⁺R⁺: right backward, straight, right forward."""
    u, t = _polar(x + r * np.sin(phi), y + r * (1 - np.cos(phi)))
    if t < -1e-12:
        return None
    v = angle_wrap(t - phi)  # dθ: +t + 0 - v = φ → v = t - φ
    if v < -1e-12:
        return None
    return [
        ('backward', -1/r, t * r),
        ('forward', 0.0, u),
        ('forward', -1/r, v * r),
    ]

# Time-reversed FFB → BBF, BFF → FBB
def _lsl_bbf(x, y, phi, r):
    segs = _lsl_bff(x, -y, -phi, r)
    return None if segs is None else [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]

def _lsr_bbf(x, y, phi, r):
    segs = _lsr_bff(x, -y, -phi, r)
    return None if segs is None else [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]

def _rsl_bbf(x, y, phi, r):
    segs = _rsl_bff(x, -y, -phi, r)
    return None if segs is None else [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]

def _rsr_bbf(x, y, phi, r):
    segs = _rsr_bff(x, -y, -phi, r)
    return None if segs is None else [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]

def _lsl_fbb(x, y, phi, r):
    segs = _lsl_ffb(x, -y, -phi, r)
    return None if segs is None else [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]

def _lsr_fbb(x, y, phi, r):
    segs = _lsr_ffb(x, -y, -phi, r)
    return None if segs is None else [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]

def _rsl_fbb(x, y, phi, r):
    segs = _rsl_ffb(x, -y, -phi, r)
    return None if segs is None else [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]

def _rsr_fbb(x, y, phi, r):
    segs = _rsr_ffb(x, -y, -phi, r)
    return None if segs is None else [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]

def _lsl_back(x, y, phi, r):
    segs = _lsl(x, -y, -phi, r)
    if segs is None:
        return None
    return [('backward', s[1], s[2]) for s in reversed(segs)]

def _lsr_back(x, y, phi, r):
    segs = _lsr(x, -y, -phi, r)
    if segs is None:
        return None
    return [('backward', s[1], s[2]) for s in reversed(segs)]

def _rsl_back(x, y, phi, r):
    segs = _rsl(x, -y, -phi, r)
    if segs is None:
        return None
    return [('backward', s[1], s[2]) for s in reversed(segs)]

def _rsr_back(x, y, phi, r):
    segs = _rsr(x, -y, -phi, r)
    if segs is None:
        return None
    return [('backward', s[1], s[2]) for s in reversed(segs)]

# CCC modes (curve-curve-curve) — LRL and RLR with forward/reverse variants

def _lrl(x, y, phi, r):
    """L⁺R⁻L⁺: left-forward, right-backward, left-forward."""
    # C1 = center of first left turn from origin
    # C3 = center of final left turn (reach goal after L turn)
    cx1, cy1 = 0.0, r
    cx3 = x - r * math.sin(phi)
    cy3 = y + r * math.cos(phi)

    dx = cx3 - cx1
    dy = cy3 - cy1
    d = math.hypot(dx, dy)
    if d > 4.0 * r + 1e-9 or d < 1e-9:
        return None

    # C2 at intersection of two circles radius=2r centered at C1, C3
    d_half = d / 2.0
    if d_half > 2.0 * r:
        return None
    h = math.sqrt(max(0.0, 4.0 * r * r - d_half * d_half))
    ux, uy = dx / d, dy / d
    px, py = -uy, ux  # perpendicular

    mid_x = cx1 + ux * d_half
    mid_y = cy1 + uy * d_half
    cx2 = mid_x + px * h
    cy2 = mid_y + py * h

    # Turn angles: from C1 to tangent point (L turn, CCW)
    alpha1 = math.atan2(cy1 - 0.0, cx1 - 0.0)  # C1 to start: (0,r)→(0,0) is downward
    # start at (0,0) heading 0, left turn center at (0,r)
    # angle from center to start: atan2(0-r, 0-0) = -pi/2
    start_ang = -math.pi / 2.0
    tp1_x = (cx1 + cx2) / 2.0  # tangent point C1-C2
    tp1_y = (cy1 + cy2) / 2.0
    t_ang = math.atan2(tp1_y - cy1, tp1_x - cx1)
    t = r * angle_wrap(t_ang - start_ang)  # CCW
    if t < 0:
        t += 2.0 * math.pi * r

    # Middle R turn (backward, CW around C2)
    # Tangent point C2-C3
    tp2_x = (cx2 + cx3) / 2.0
    tp2_y = (cy2 + cy3) / 2.0
    ang_to_tp1 = math.atan2(tp1_y - cy2, tp1_x - cx2)
    ang_to_tp2 = math.atan2(tp2_y - cy2, tp2_x - cx2)
    u = r * angle_wrap(ang_to_tp1 - ang_to_tp2)  # CW (R turn), angle from tp1 to tp2
    if u < 0:
        u += 2.0 * math.pi * r

    # Final L turn (forward, CCW around C3)
    goal_ang = math.atan2(y - cy3, x - cx3)
    ang_from_tp2 = math.atan2(tp2_y - cy3, tp2_x - cx3)
    v = r * angle_wrap(goal_ang - ang_from_tp2)
    if v < 0:
        v += 2.0 * math.pi * r

    if t < -1e-12 or u < -1e-12 or v < -1e-12:
        return None

    return [
        ('forward', 1.0 / r, t),
        ('backward', -1.0 / r, u),
        ('forward', 1.0 / r, v),
    ]


def _rlr(x, y, phi, r):
    """R⁺L⁻R⁺: right-forward, left-backward, right-forward."""
    segs = _lrl(x, -y, -phi, r)
    if segs is None:
        return None
    return [(_flip_dir(s[0]), -s[1], s[2]) for s in segs]


def _lrl_back(x, y, phi, r):
    """Time-reversed LRL (all directions flipped, segments reversed)."""
    segs = _lrl(x, -y, -phi, r)
    if segs is None:
        return None
    return [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]


def _rlr_back(x, y, phi, r):
    """Time-reversed RLR."""
    segs = _rlr(x, -y, -phi, r)
    if segs is None:
        return None
    return [(_flip_dir(s[0]), s[1], s[2]) for s in reversed(segs)]


def _flip_dir(direction):
    return 'backward' if direction == 'forward' else 'forward'


_RS_MODES = [
    # CSC all-forward (C⁺S⁺C⁺)
    _lsl, _lsr, _rsl, _rsr,
    # CSC all-backward (C⁻S⁻C⁻) via time-reversal
    _lsl_back, _lsr_back, _rsl_back, _rsr_back,
    # CSC mixed: overshoot & back in (C⁺S⁺C⁻)
    _lsl_ffb, _lsr_ffb, _rsl_ffb, _rsr_ffb,
    # CSC mixed: back out & drive (C⁻S⁺C⁺)
    _lsl_bff, _lsr_bff, _rsl_bff, _rsr_bff,
    # CSC mixed: backward-backward-forward (C⁻S⁻C⁺, time-rev of FFB)
    _lsl_bbf, _lsr_bbf, _rsl_bbf, _rsr_bbf,
    # CSC mixed: forward-backward-backward (C⁺S⁻C⁻, time-rev of BFF)
    _lsl_fbb, _lsr_fbb, _rsl_fbb, _rsr_fbb,
    # CCC (C|C|C)
    _lrl, _rlr, _lrl_back, _rlr_back,
]


def reeds_shepp_path_pose(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    radius: float,
    num_points: int = 20,
) -> tuple[np.ndarray, list[dict]]:
    """Like reeds_shepp_path but returns N×3 [x, y, theta] with theta from
    vehicle integration (not XY differencing).  Preserves motion direction at
    cusp points where Δx,Δy≈0 would cause atan2 noise.
    """
    sx, sy, st = float(start[0]), float(start[1]), float(start[2])
    gx, gy, gt = float(goal[0]), float(goal[1]), float(goal[2])
    r = float(radius)

    cos_st = np.cos(st)
    sin_st = np.sin(st)
    dx = gx - sx
    dy = gy - sy
    gx_rel = dx * cos_st + dy * sin_st
    gy_rel = -dx * sin_st + dy * cos_st
    gt_rel = angle_wrap(gt - st)

    segments_rel = _solve_rs(gx_rel, gy_rel, gt_rel, r)
    if segments_rel is None:
        return None, []

    segments = []
    for seg in segments_rel:
        direction, curvature, length = seg
        if curvature > 0:
            direction_side = 'left'
        elif curvature < 0:
            direction_side = 'right'
        else:
            direction_side = 'straight'
        segments.append({
            'type': direction,
            'length': length,
            'curvature': curvature,
            'direction': direction_side,
        })

    pose = _sample_segments_pose(sx, sy, st, segments_rel, num_points)
    return pose, segments


def _sample_segments_pose(x0, y0, theta0, segments, num_points):
    """Sample N×3 [x, y, theta] from segments — vectorized per-segment."""
    lengths = np.array([s[2] for s in segments], dtype=np.float64)
    curvatures = np.array([s[1] for s in segments], dtype=np.float64)
    directions = np.array([1.0 if s[0] == 'forward' else -1.0 for s in segments],
                          dtype=np.float64)
    total = lengths.sum()
    if total < 1e-6:
        return np.column_stack([
            np.full(num_points, x0), np.full(num_points, y0),
            np.full(num_points, theta0),
        ])

    # Cumulative segment starts
    cumlen = np.concatenate([[0.0], np.cumsum(lengths)])
    sample_dists = np.linspace(0, total, num_points)
    seg_idx = np.clip(np.searchsorted(cumlen, sample_dists, side='right') - 1, 0, len(segments) - 1)

    # Pre-compute start pose of each segment by integrating all previous
    seg_starts = np.zeros((len(segments), 3), dtype=np.float64)
    x, y, t = float(x0), float(y0), float(theta0)
    for i in range(len(segments)):
        seg_starts[i] = [x, y, t]
        x, y, t = _integrate_segment(x, y, t, curvatures[i], directions[i], lengths[i])

    # Per-point: distance from segment start
    pose = np.empty((num_points, 3), dtype=np.float64)
    for s in range(len(segments)):
        mask = seg_idx == s
        if not mask.any():
            continue
        n_pts = mask.sum()
        ds = sample_dists[mask] - cumlen[s]  # distances from segment start
        sx, sy, st = seg_starts[s]
        k, d = curvatures[s], directions[s]

        if abs(k) < 1e-9:
            pose[mask, 0] = sx + d * ds * np.cos(st)
            pose[mask, 1] = sy + d * ds * np.sin(st)
            pose[mask, 2] = st
        else:
            r = 1.0 / abs(k)
            dtheta = d * ds * k
            thetas = st + dtheta
            if k > 0:
                cx = sx - r * np.sin(st)
                cy = sy + r * np.cos(st)
                pose[mask, 0] = cx + r * np.sin(thetas)
                pose[mask, 1] = cy - r * np.cos(thetas)
            else:
                cx = sx + r * np.sin(st)
                cy = sy - r * np.cos(st)
                pose[mask, 0] = cx - r * np.sin(thetas)
                pose[mask, 1] = cy + r * np.cos(thetas)
            pose[mask, 2] = thetas

    return pose


def reeds_shepp_path_first_valid(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    radius: float,
    num_points: int = 20,
    use_reverse: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Lazy RS: return the first valid mode (no collision check here — just
    the first mode that produces a geometrically-valid path).

    Unlike ``reeds_shepp_path`` which computes all 8 modes and picks the
    shortest, this iterates modes in a fixed order and returns the first one
    that succeeds.  This saves ~87% of mode computation when the analytic
    expansion is used as a search shortcut (any valid path works).

    Args:
        start: (x, y, theta) start pose in world frame.
        goal: (x, y, theta) goal pose in world frame.
        radius: minimum turning radius in meters.
        num_points: number of waypoints to sample.
        use_reverse: if False, skip backward modes.

    Returns:
        path_waypoints: (num_points, 2) sampled points in world coords, or None.
        segments: list of segment dicts.
    """
    sx, sy, st = float(start[0]), float(start[1]), float(start[2])
    gx, gy, gt = float(goal[0]), float(goal[1]), float(goal[2])
    r = float(radius)

    cos_st = np.cos(st)
    sin_st = np.sin(st)
    dx = gx - sx
    dy = gy - sy
    gx_rel = dx * cos_st + dy * sin_st
    gy_rel = -dx * sin_st + dy * cos_st
    gt_rel = angle_wrap(gt - st)

    # Iterate modes, return first geometrically-valid one
    modes = list(_RS_MODES)
    for mode_fn in modes:
        segs = mode_fn(gx_rel, gy_rel, gt_rel, r)
        if segs is None:
            continue
        # Check reverse constraint
        if not use_reverse and any(s[0] == 'backward' for s in segs):
            continue

        # Build segment dicts
        segments = []
        for seg in segs:
            direction, curvature, length = seg
            if curvature > 0:
                direction_side = 'left'
            elif curvature < 0:
                direction_side = 'right'
            else:
                direction_side = 'straight'
            segments.append({
                'type': direction,
                'length': length,
                'curvature': curvature,
                'direction': direction_side,
            })

        path = _sample_segments(sx, sy, st, segs, num_points)
        return path, segments

    return None, []


def _sample_segments(x0, y0, theta0, segments, num_points):
    total_length = sum(s[2] for s in segments)
    if total_length < 1e-6:
        return np.tile(np.array([[x0, y0]]), (num_points, 1))

    sample_distances = np.linspace(0, total_length, num_points)
    points = np.empty((num_points, 2))

    x, y, theta = float(x0), float(y0), float(theta0)
    seg_idx = 0
    seg_start = 0.0
    seg_length = abs(segments[0][2])
    curvature = segments[0][1]
    direction = 1.0 if segments[0][0] == 'forward' else -1.0

    for i, dist in enumerate(sample_distances):
        while dist > seg_start + seg_length + 1e-9:
            x, y, theta = _integrate_segment(x, y, theta, curvature, direction, seg_length)
            seg_start += seg_length
            seg_idx += 1
            if seg_idx >= len(segments):
                break
            seg_length = abs(segments[seg_idx][2])
            curvature = segments[seg_idx][1]
            direction = 1.0 if segments[seg_idx][0] == 'forward' else -1.0

        remaining = dist - seg_start
        px, py, _ = _integrate_segment(x, y, theta, curvature, direction, remaining)
        points[i] = [px, py]

    return points


def _integrate_segment(x, y, theta, curvature, direction, length):
    if abs(curvature) < 1e-9:
        return (
            x + direction * length * np.cos(theta),
            y + direction * length * np.sin(theta),
            theta,
        )
    r = 1.0 / abs(curvature)
    # dtheta = direction * length * curvature: 物理学正确。
    # curvature 是路径曲率（+1/r=左转路径），方向不影响路径形状。
    dtheta = direction * length * curvature
    if curvature > 0:  # 左转路径 — 圆心在车体左侧
        cx = x - r * np.sin(theta)
        cy = y + r * np.cos(theta)
    else:              # 右转路径 — 圆心在车体右侧
        cx = x + r * np.sin(theta)
        cy = y - r * np.cos(theta)
    new_theta = theta + dtheta
    if curvature > 0:
        nx = cx + r * np.sin(new_theta)
        ny = cy - r * np.cos(new_theta)
    else:
        nx = cx - r * np.sin(new_theta)
        ny = cy + r * np.cos(new_theta)
    return (nx, ny, new_theta)
