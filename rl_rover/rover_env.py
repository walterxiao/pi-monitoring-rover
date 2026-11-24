#!/usr/bin/env python3
"""Gymnasium environment for the Pi monitoring rover.

Task: drive to a goal point in a 2D room without hitting anything.
Trained with RL (PPO) in simulation on the Mac, then deployed to the
Pi as an ONNX policy — the Microduck pattern (train in sim, run the
trained model on the Pi; the Pi never trains).

State (27 dims): 24 LD19 sectors (min distance per 15 deg, /12 m)
  + goal bearing (sin, cos) + goal distance (/5 m).
Actions (discrete): 0 = forward, 1 = turn left, 2 = turn right, 3 = stop.
Dynamics match the real rover: 0.3 m/s forward, 120 deg/s turn
(180 deg in 1.5 s, measured), heading is a compass bearing
(0 = north/+y, clockwise-positive, left turn decreases it).
Reward: +100 goal, -100 collision, -0.1/step, +10 x progress
toward the goal, a side-margin term (-1.0 x margin violation)
that keeps enough clearance on the rover's tighter side (no need
to be exactly centered, but in a narrow path any drift off-center
violates the margin, so it stays centered there), and small
penalties for turning (-0.05) and stopping (-0.05) so the shortest
time wins — the policy prefers to drive straight and only turns
when it needs to, instead of stop-and-turn.

Domain randomization per episode (sim-to-real): random start pose,
random goal, 0-3 box obstacles, LiDAR noise + dropouts, and +/-10%
speed / turn-rate jitter, so the policy is robust instead of
memorizing one room.
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- room / rover constants (match the real rover) -------------------------
ROOM_W = 4.0          # m, x in [-2, 2]
ROOM_H = 3.0          # m, y in [-1.5, 1.5]
MAX_RANGE_M = 12.0    # LD19 max range
N_RAYS = 72           # 5 deg apart; 3 rays per 15 deg sector
RAY_STEP_DEG = 360.0 / N_RAYS
N_SECTORS = 24        # 15 deg per sector
DT = 0.1              # s per step (10 Hz)
FWD_SPEED = 0.3       # m/s (FORWARD_SPEED_M_S in rover_server.py)
TURN_RATE = 120.0     # deg/s (180 deg in 1.5 s, measured)
COLLISION_R = 0.20    # m — body radius; any ray closer = hit
GOAL_R = 0.25         # m — within this = goal reached
MAX_STEPS = 300       # 30 s per episode
GOAL_REWARD = 100.0
COLLISION_PENALTY = -100.0
STEP_PENALTY = -0.1
PROGRESS_SCALE = 10.0
CENTER_SCALE = 1.0  # per-step penalty for violating the side
# margin: the rover should keep enough clearance on its tighter
# side. It does NOT need to be exactly centered — as long as the
# nearer side has SIDE_MARGIN_M clearance, there is no penalty.
# In a narrow path any drift off-center shrinks the nearer
# clearance below the margin, so the penalty pushes the rover
# back to center.
SIDE_MARGIN_M = 0.35  # minimum clearance on the tighter side (m);
# the rover's body radius is 0.20 m, so this leaves 0.15 m of
# edge clearance.
TURN_PENALTY = -0.05  # per turn action (left/right): shortest time
# wins, so turning costs a little extra — the policy prefers to
# drive straight and only turns when it needs to (to avoid an
# obstacle or head to the goal), instead of stop-and-turn.
STOP_PENALTY = -0.05  # per stop action: don't sit still when
# you could be driving (the goal is not reached by waiting).
LIDAR_NOISE_M = 0.02
LIDAR_DROPOUT_P = 0.01
WALL_MARGIN = 0.4     # start/goal/obstacle margin from the walls


class RoverEnv(gym.Env):
    """Drive to a goal in a 2D room, guided by a 360 deg LiDAR scan."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, room_w=ROOM_W, room_h=ROOM_H, max_steps=MAX_STEPS,
                 n_obstacles=3, seed=None):
        super().__init__()
        self.room_w = room_w
        self.room_h = room_h
        self.max_steps = max_steps
        self.max_obstacles = n_obstacles
        # 27-dim obs: 24 lidar sectors [0,1] + sin/cos bearing [-1,1]
        # + goal distance [0,1].
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_SECTORS + 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # fwd / left / right / stop
        self._rng = np.random.default_rng(seed)
        self.x = self.y = self.heading = 0.0
        self.gx = self.gy = 0.0
        self.obstacles = []  # list of (x0, y0, x1, y1)
        self.speed = FWD_SPEED
        self.turn_rate = TURN_RATE
        self.steps = 0
        self.prev_goal_dist = 0.0
        self._fig = None

    # --- geometry ----------------------------------------------------------
    def _scan(self):
        """72-ray LiDAR scan (5 deg apart, ray 0 = straight ahead).

        Returns distances in metres, one per ray, ray-cast against the
        room walls and the box obstacles (vectorized over the rays).
        """
        angs = np.deg2rad(self.heading + np.arange(N_RAYS) * RAY_STEP_DEG)
        dx = np.sin(angs)
        dy = np.cos(angs)
        dist = np.full(N_RAYS, MAX_RANGE_M)
        hw, hh = self.room_w / 2.0, self.room_h / 2.0
        # x-walls
        for wall_x in (hw, -hw):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wall_x - self.x) / dx
            hit_y = self.y + t * dy
            valid = (t > 0) & (np.abs(hit_y) <= hh)
            dist = np.minimum(dist, np.where(valid, t, np.inf))
        # y-walls
        for wall_y in (hh, -hh):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wall_y - self.y) / dy
            hit_x = self.x + t * dx
            valid = (t > 0) & (np.abs(hit_x) <= hw)
            dist = np.minimum(dist, np.where(valid, t, np.inf))
        # box obstacles (slab method, vectorized)
        for (x0, y0, x1, y1) in self.obstacles:
            dist = np.minimum(dist, self._ray_aabb(dx, dy, x0, y0, x1, y1))
        return dist

    def _ray_aabb(self, dx, dy, x0, y0, x1, y1):
        """Entry distance of each ray into an axis-aligned box (inf if miss)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            tx1 = (x0 - self.x) / dx
            tx2 = (x1 - self.x) / dx
            ty1 = (y0 - self.y) / dy
            ty2 = (y1 - self.y) / dy
        tx_lo = np.minimum(tx1, tx2)
        tx_hi = np.maximum(tx1, tx2)
        ty_lo = np.minimum(ty1, ty2)
        ty_hi = np.maximum(ty1, ty2)
        tmin = np.maximum(tx_lo, ty_lo)
        tmax = np.minimum(tx_hi, ty_hi)
        hit = (tmin <= tmax) & (tmax > 0) & np.isfinite(tmin)
        # Origin inside the box -> tmin < 0; clamp to 0 (contact).
        return np.where(hit, np.maximum(tmin, 0.0), np.inf)

    def _noisy_scan(self):
        """The scan the policy sees: real LD19 noise + dropouts."""
        d = self._scan()
        d = d + self._rng.normal(0.0, LIDAR_NOISE_M, size=d.shape)
        drop = self._rng.random(size=d.shape) < LIDAR_DROPOUT_P
        d = np.where(drop, MAX_RANGE_M, d)
        return np.clip(d, 0.0, MAX_RANGE_M)

    def _sectors(self, dist):
        """Downsample the ray distances to 24 sectors (min per sector)."""
        return dist.reshape(N_SECTORS, -1).min(axis=1)

    def _goal_dist(self):
        return math.hypot(self.gx - self.x, self.gy - self.y)

    def _margin_violation(self, dist):
        """How badly the rover violates its side margin, from the
        true 72-ray scan (ray 0 = straight ahead). Left = sectors
        16-20 (240-300 deg, centered on the rover's left); right =
        sectors 4-8 (60-120 deg, centered on its right). Returns
        0.0 when the nearer side has at least SIDE_MARGIN_M
        clearance (good enough — no need to be exactly centered),
        ramping linearly to 1.0 as the nearer clearance shrinks
        to zero. In a narrow path any drift off-center shrinks the
        nearer clearance below the margin, so the penalty pushes
        the rover back to center."""
        sectors = self._sectors(dist)
        left_d = float(sectors[16:21].min())
        right_d = float(sectors[4:9].min())
        clearance = min(left_d, right_d)
        if clearance >= SIDE_MARGIN_M:
            return 0.0
        return (SIDE_MARGIN_M - clearance) / SIDE_MARGIN_M

    def _obs(self):
        sectors = self._sectors(self._noisy_scan()) / MAX_RANGE_M
        # Bearing to the goal, relative to the rover's heading.
        bearing = math.degrees(math.atan2(self.gx - self.x,
                                          self.gy - self.y))
        rel = math.radians((bearing - self.heading + 180.0) % 360.0 - 180.0)
        gd = self._goal_dist() / 5.0
        obs = np.concatenate([sectors,
                              [math.sin(rel), math.cos(rel),
                               min(gd, 1.0)]]).astype(np.float32)
        return obs

    # --- episode setup ------------------------------------------------------
    def _random_point(self, margin):
        hw, hh = self.room_w / 2.0, self.room_h / 2.0
        return (self._rng.uniform(-hw + margin, hw - margin),
                self._rng.uniform(-hh + margin, hh - margin))

    def _clear_of_obstacles(self, px, py, margin):
        for (x0, y0, x1, y1) in self.obstacles:
            if x0 - margin < px < x1 + margin and y0 - margin < py < y1 + margin:
                return False
        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Random obstacles first (0..max), then a start and goal clear of them.
        n = int(self._rng.integers(0, self.max_obstacles + 1))
        self.obstacles = []
        for _ in range(n):
            for _try in range(20):
                cx, cy = self._random_point(WALL_MARGIN + 0.3)
                w = self._rng.uniform(0.2, 0.6)
                h = self._rng.uniform(0.2, 0.6)
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                # Keep the box inside the room.
                hw, hh = self.room_w / 2.0, self.room_h / 2.0
                if (box[0] < -hw + WALL_MARGIN or box[2] > hw - WALL_MARGIN
                        or box[1] < -hh + WALL_MARGIN
                        or box[3] > hh - WALL_MARGIN):
                    continue
                self.obstacles.append(box)
                break
        for _try in range(50):
            self.x, self.y = self._random_point(WALL_MARGIN)
            if self._clear_of_obstacles(self.x, self.y, 0.3):
                break
        self.heading = float(self._rng.uniform(0.0, 360.0))
        for _try in range(50):
            self.gx, self.gy = self._random_point(WALL_MARGIN)
            if (self._goal_dist() >= 0.8
                    and self._clear_of_obstacles(self.gx, self.gy, 0.4)):
                break
        # Per-episode dynamics jitter (sim-to-real).
        self.speed = FWD_SPEED * float(self._rng.uniform(0.9, 1.1))
        self.turn_rate = TURN_RATE * float(self._rng.uniform(0.9, 1.1))
        self.steps = 0
        self.prev_goal_dist = self._goal_dist()
        return self._obs(), {"goal_dist": self.prev_goal_dist}

    # --- step ---------------------------------------------------------------
    def step(self, action):
        a = int(action)
        if a == 0:  # forward
            h = math.radians(self.heading)
            self.x += self.speed * DT * math.sin(h)
            self.y += self.speed * DT * math.cos(h)
        elif a == 1:  # turn left (heading decreases: compass bearing)
            self.heading = (self.heading - self.turn_rate * DT) % 360.0
        elif a == 2:  # turn right
            self.heading = (self.heading + self.turn_rate * DT) % 360.0
        # a == 3: stop (no movement)
        self.steps += 1
        dist = self._scan()
        min_d = float(dist.min())
        gd = self._goal_dist()
        # Side margin: keep enough clearance on the tighter side.
        # No penalty when the margin is met (no need to be exactly
        # centered); in a narrow path any drift off-center shrinks
        # the nearer clearance below the margin, so the penalty
        # pushes the rover back to center.
        margin_v = self._margin_violation(dist)
        reward = (STEP_PENALTY
                  + (self.prev_goal_dist - gd) * PROGRESS_SCALE
                  - CENTER_SCALE * margin_v)
        # Shortest time wins: turning and stopping cost a little
        # extra, so the policy prefers to drive straight and only
        # turns when it needs to — less stop-and-turn.
        if a == 1 or a == 2:  # turn left / right
            reward += TURN_PENALTY
        elif a == 3:  # stop
            reward += STOP_PENALTY
        self.prev_goal_dist = gd
        terminated = False
        info = {"goal_dist": gd, "min_lidar": min_d, "steps": self.steps,
                "margin_violation": margin_v}
        if min_d < COLLISION_R:
            reward += COLLISION_PENALTY
            terminated = True
            info["collision"] = True
        elif gd <= GOAL_R:
            reward += GOAL_REWARD
            terminated = True
            info["success"] = True
        truncated = (not terminated) and (self.steps >= self.max_steps)
        return self._obs(), float(reward), terminated, truncated, info

    # --- render ---------------------------------------------------------------
    def render(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle
        except ImportError:
            return None
        hw, hh = self.room_w / 2.0, self.room_h / 2.0
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.set_xlim(-hw - 0.2, hw + 0.2)
        ax.set_ylim(-hh - 0.2, hh + 0.2)
        ax.set_aspect("equal")
        ax.add_patch(Rectangle((-hw, -hh), self.room_w, self.room_h,
                               fill=False, edgecolor="black", lw=2))
        for (x0, y0, x1, y1) in self.obstacles:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor="gray", edgecolor="black"))
        ax.add_patch(Circle((self.gx, self.gy), GOAL_R,
                            facecolor="green", alpha=0.4, label="goal"))
        ax.plot(self.gx, self.gy, "g*", ms=14)
        # LiDAR rays (every 5th ray, for readability).
        d = self._scan()
        for i in range(0, N_RAYS, 5):
            a = math.radians(self.heading + i * RAY_STEP_DEG)
            ax.plot([self.x, self.x + d[i] * math.sin(a)],
                    [self.y, self.y + d[i] * math.cos(a)],
                    "c-", lw=0.5, alpha=0.5)
        h = math.radians(self.heading)
        ax.plot([self.x, self.x + 0.3 * math.sin(h)],
                [self.y, self.y + 0.3 * math.cos(h)], "r-", lw=2)
        ax.plot(self.x, self.y, "bo", ms=10)
        ax.set_title(f"step {self.steps}  goal_dist {self._goal_dist():.2f} m")
        ax.set_xlabel("x (m, east)"); ax.set_ylabel("y (m, north)")
        return fig


if __name__ == "__main__":
    # Quick smoke test: random policy for a few episodes.
    env = RoverEnv()
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        total = 0.0
        for _ in range(MAX_STEPS):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            total += r
            if term or trunc:
                break
        print(f"ep {ep}: steps={info['steps']} reward={total:.1f} "
              f"success={info.get('success', False)} "
              f"collision={info.get('collision', False)} "
              f"goal_dist={info['goal_dist']:.2f}")
