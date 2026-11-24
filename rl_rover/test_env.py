#!/usr/bin/env python3
"""Verify rover_env.py:
1. Vectorized scan matches the scalar room_scan (walls only) at the
   same 72 ray angles (every 5 deg).
2. Box ray-casting matches a scalar ray-segment reference.
3. Env API: reset/step shapes, reward signs, collision ends episode,
   goal ends episode, heading convention (left decreases heading),
   forward moves north at heading 0, obstacles are respected.
"""
import math
import sys

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/goals/mobile-pi-monitoring-rover/files/rl_rover")
from rover_env import RoverEnv, N_RAYS, RAY_STEP_DEG, MAX_RANGE_M

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# --- scalar reference: walls only (same math as test_escape.py) ------------
def room_scan_scalar(rx, ry, heading_deg, n_rays=72, step_deg=5.0):
    bins = []
    for b in range(n_rays):
        a = math.radians((heading_deg + b * step_deg) % 360)
        dx, dy = math.sin(a), math.cos(a)
        best = None
        for wall_d, wall_axis in ((2.0 - rx, "x+"), (-2.0 - rx, "x-"),
                                  (1.5 - ry, "y+"), (-1.5 - ry, "y-")):
            if wall_axis.startswith("x"):
                if abs(dx) < 1e-9:
                    continue
                t = wall_d / dx
                if t <= 0:
                    continue
                if abs(ry + t * dy) <= 1.5:
                    best = t if best is None else min(best, t)
            else:
                if abs(dy) < 1e-9:
                    continue
                t = wall_d / dy
                if t <= 0:
                    continue
                if abs(rx + t * dx) <= 2.0:
                    best = t if best is None else min(best, t)
        bins.append(best if best is not None else MAX_RANGE_M)
    return np.array(bins)


def ray_segment(ox, oy, dx, dy, x1, y1, x2, y2):
    """Scalar ray vs segment intersection distance, or None."""
    ex, ey = x2 - x1, y2 - y1
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
    if t > 0 and 0.0 <= u <= 1.0:
        return t
    return None


def box_scan_scalar(rx, ry, heading_deg, box, n_rays=72, step_deg=5.0):
    x0, y0, x1, y1 = box
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
             ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    out = []
    for b in range(n_rays):
        a = math.radians((heading_deg + b * step_deg) % 360)
        dx, dy = math.sin(a), math.cos(a)
        best = None
        for (p, q) in edges:
            t = ray_segment(rx, ry, dx, dy, p[0], p[1], q[0], q[1])
            if t is not None:
                best = t if best is None else min(best, t)
        out.append(best if best is not None else np.inf)
    return np.array(out)


env = RoverEnv()

# 1. Walls-only scan matches the scalar reference at several poses.
for (rx, ry, h) in [(0.0, 0.0, 0.0), (-1.5, 1.1, 0.0), (1.2, -0.8, 137.0),
                    (0.5, 0.5, 270.0), (-1.9, -1.4, 45.0)]:
    env.x, env.y, env.heading = rx, ry, h
    env.obstacles = []
    got = env._scan()
    want = room_scan_scalar(rx, ry, h)
    check(f"walls scan matches scalar at ({rx},{ry},{h})",
          np.allclose(got, want, atol=1e-9),
          f"max err {np.abs(got - want).max():.2e}")

# 2. Box scan matches the scalar segment reference.
box = (0.5, -0.5, 1.1, 0.3)
for (rx, ry, h) in [(0.0, 0.0, 0.0), (-1.0, 0.5, 90.0), (0.0, 1.0, 180.0)]:
    env.x, env.y, env.heading = rx, ry, h
    env.obstacles = [box]
    got = env._scan()
    want_walls = room_scan_scalar(rx, ry, h)
    want_box = box_scan_scalar(rx, ry, h, box)
    want = np.minimum(want_walls, want_box)
    check(f"box scan matches scalar at ({rx},{ry},{h})",
          np.allclose(got, want, atol=1e-9),
          f"max err {np.abs(got - want).max():.2e}")

# 3. Env API + dynamics.
obs, info = env.reset(seed=42)
check("obs shape is 27", obs.shape == (27,), f"got {obs.shape}")
check("obs in [-1, 1]", bool(np.all(obs >= -1.0) and np.all(obs <= 1.0)),
      f"range {obs.min():.2f}..{obs.max():.2f}")
check("action space is Discrete(4)", env.action_space.n == 4)

# Forward at heading 0 moves north (+y) by speed*dt.
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.gx, env.gy = 1.5, 1.0  # goal far away so no early termination
env.obstacles = []
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, _, term, trunc, _ = env.step(0)
check("forward at heading 0 moves north",
      abs(env.x - 0.0) < 1e-9 and env.y > 0.0,
      f"pose=({env.x:.3f},{env.y:.3f})")
check("forward step distance is speed*dt",
      abs(env.y - env.speed * 0.1) < 1e-9, f"y={env.y:.4f}")

# Turn left decreases heading; turn right increases it.
env.heading = 90.0
env.step(1)
check("turn left decreases heading", abs(env.heading - (90.0 - env.turn_rate * 0.1)) < 1e-9,
      f"heading={env.heading:.1f}")
env.step(2)
check("turn right increases heading", abs(env.heading - 90.0) < 1e-9,
      f"heading={env.heading:.1f}")

# Moving toward the goal gives positive reward; moving away negative.
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.gx, env.gy = 0.0, 1.5  # goal straight north
env.obstacles = []
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, r_fwd, _, _, _ = env.step(0)
check("moving toward goal gives positive reward", r_fwd > 0, f"r={r_fwd:.3f}")
env.heading = 180.0  # face south, away from the goal
env.prev_goal_dist = env._goal_dist()
_, r_away, _, _, _ = env.step(0)
check("moving away from goal gives negative reward", r_away < 0, f"r={r_away:.3f}")

# Collision ends the episode with a big negative reward.
# Start 0.20 m from the north wall; one 0.03 m step forward puts the
# wall 0.17 m away, inside the 0.20 m body radius.
env.x, env.y, env.heading = 0.0, 1.30, 0.0
env.gx, env.gy = 0.0, -1.0
env.obstacles = []
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, r_col, term, _, info = env.step(0)  # drive into the wall
check("collision terminates the episode", term is True, f"term={term}")
check("collision reward is strongly negative", r_col < -50, f"r={r_col:.1f}")
check("collision flagged in info", info.get("collision") is True)

# Reaching the goal ends the episode with a big positive reward.
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.gx, env.gy = 0.0, 0.20  # goal 0.2 m ahead (inside GOAL_R after 1 step)
env.obstacles = []
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, r_goal, term, _, info = env.step(0)
check("goal terminates the episode", term is True, f"term={term}")
check("goal reward is strongly positive", r_goal > 50, f"r={r_goal:.1f}")
check("success flagged in info", info.get("success") is True)

# --- 4. Side margin: enough clearance, no need for exact center ------------
# Centered in the room (equal clearance both sides) -> violation 0.
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.obstacles = []
dist = env._scan()
v_center = env._margin_violation(dist)
check("centered in room -> margin violation 0",
      v_center == 0.0, f"v={v_center:.3f}")
# Off-center but with enough margin on both sides (x = -1.0: left
# 1.0 m, right 3.0 m) -> still 0. No need to be exactly centered.
env.x, env.y, env.heading = -1.0, 0.0, 0.0
dist = env._scan()
v_off = env._margin_violation(dist)
check("off-center with margin on both sides -> violation 0",
      v_off == 0.0, f"v={v_off:.3f}")
# Hugging the left wall (x = -1.8, left side ~0.2 m) -> high violation.
env.x, env.y, env.heading = -1.8, 0.0, 0.0
dist = env._scan()
v_hug = env._margin_violation(dist)
check("hugging left wall -> margin violation high",
      v_hug > 0.4, f"v={v_hug:.3f}")
# Hugging the right wall mirrors it.
env.x, env.y, env.heading = 1.8, 0.0, 0.0
dist = env._scan()
v_hug_r = env._margin_violation(dist)
check("hugging right wall -> margin violation high",
      v_hug_r > 0.4, f"v={v_hug_r:.3f}")
# Narrow path: a 0.7 m corridor (walls at x = +/-0.35). Centered
# the rover just meets the margin; drifting off-center violates it.
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.obstacles = [(-2.0, -1.5, -0.35, 1.5), (0.35, -1.5, 2.0, 1.5)]
dist = env._scan()
v_narrow_center = env._margin_violation(dist)
env.x, env.y, env.heading = 0.15, 0.0, 0.0  # drifted 0.15 m right
dist = env._scan()
v_narrow_drift = env._margin_violation(dist)
check("narrow path: centered meets margin, drift violates it",
      v_narrow_center < v_narrow_drift,
      f"centered v={v_narrow_center:.3f}, drifted v={v_narrow_drift:.3f}")
# The reward penalizes margin violation: same forward step, same
# goal progress, but with margin vs hugging — the margin step scores higher.
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.gx, env.gy = 0.0, 1.5
env.obstacles = []
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, r_margin, _, _, info_m = env.step(0)
env.x, env.y, env.heading = -1.8, 0.0, 0.0
env.gx, env.gy = -1.8, 1.5  # same northward progress, hugging the wall
env.obstacles = []
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, r_hug, _, _, info_h = env.step(0)
check("step with margin scores higher than wall-hugging step",
      r_margin > r_hug,
      f"r_margin={r_margin:.3f}, r_hug={r_hug:.3f}")
check("margin violation is reported in info",
      "margin_violation" in info_m and "margin_violation" in info_h)

# --- 5. Shortest time wins: turning and stopping cost extra ----------------
# Same position, same goal: a forward step (toward the goal) scores
# higher than a turn step (no progress + turn penalty), and a turn
# scores higher than a stop (turn penalty = stop penalty, but the
# turn at least reorients toward the goal — here we just check the
# penalties are applied, not the full policy choice).
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.gx, env.gy = 0.0, 1.5
env.obstacles = []
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, r_fwd2, _, _, _ = env.step(0)  # forward toward the goal
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, r_turn, _, _, _ = env.step(1)  # turn left (no progress + penalty)
env.x, env.y, env.heading = 0.0, 0.0, 0.0
env.prev_goal_dist = env._goal_dist()
env.steps = 0
_, r_stop, _, _, _ = env.step(3)  # stop (no progress + penalty)
check("forward toward goal scores higher than a turn",
      r_fwd2 > r_turn, f"r_fwd={r_fwd2:.3f}, r_turn={r_turn:.3f}")
check("turn costs more than the step penalty alone",
      r_turn < -0.1, f"r_turn={r_turn:.3f} (step penalty is -0.1)")
check("stop costs more than the step penalty alone",
      r_stop < -0.1, f"r_stop={r_stop:.3f} (step penalty is -0.1)")
o1, _ = env.reset(seed=1)
o2, _ = env.reset(seed=2)
check("different seeds give different episodes",
      not np.allclose(o1, o2), "obs identical across seeds")
# A random policy should sometimes succeed and sometimes collide over many
# episodes (sanity: the env is neither impossible nor trivial).
rng = np.random.default_rng(0)
succ = coll = 0
for ep in range(200):
    env.reset(seed=1000 + ep)
    for _ in range(300):
        _, _, term, trunc, info = env.step(int(rng.integers(0, 4)))
        if term or trunc:
            break
    if info.get("success"):
        succ += 1
    if info.get("collision"):
        coll += 1
check("random policy sometimes succeeds (>0)", succ > 0, f"succ={succ}/200")
check("random policy sometimes collides (>0)", coll > 0, f"coll={coll}/200")
print(f"  (random policy: {succ} successes, {coll} collisions in 200 eps)")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
