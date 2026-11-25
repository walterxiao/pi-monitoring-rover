#!/usr/bin/env python3
"""Verify the tour mode in rover_server.py.

1. tour_goal_direction() picks the most open corridor (not a wall).
2. tour_pick_goal() sets a goal out in that direction.
3. In tour mode the policy loop keeps driving: reaching a virtual
   goal repicks a new one instead of stopping.
4. Live HTTP endpoints: /tour/on, /tour/ping, /tour/off.
5. Mutual exclusion: /policy/on turns the tour off (and clears the
   virtual goal); /tour/on turns the policy's fixed goal off.
"""
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/home/hatch/workspace/goals/mobile-pi-monitoring-rover/files/rl_rover")
sys.path.insert(0, "/home/hatch/workspace/pi-monitoring-rover")

import numpy as np
import rover_server as rs

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


SMOKE_ONNX = ("/home/hatch/workspace/goals/mobile-pi-monitoring-rover/"
              "files/rl_rover/runs/smoke/rover_policy.onnx")
rs.POLICY_ONNX_PATH = SMOKE_ONNX
rs.MOTORS_OK = True  # pretend the motors are present (no HAT on this VM)

# Fake the TF-Luna as clear so only the LD19 matters.
rs.lidar_reading = lambda: (8.0, 100)


def set_ld19(bins):
    with rs.LD19_LOCK:
        rs.LD19["online"] = True
        rs.LD19["updated"] = time.time()
        rs.LD19["bins"] = list(bins)


def set_pose(x, y, heading):
    with rs.POSE_LOCK:
        rs.POSE["x"] = x
        rs.POSE["y"] = y
        rs.POSE["heading"] = heading


# --- 1. Direction picking -------------------------------------------------
print("== tour_goal_direction picks open space ==")
# Wall dead ahead + left (bins 270-30), open on the right/back.
bins = [12.0] * 360
for b in list(range(270, 360)) + list(range(0, 31)):
    bins[b] = 0.4
set_ld19(bins)
res = rs.tour_goal_direction()
check("picks a direction", res is not None)
if res is not None:
    ang, mind = res
    # The open side is bins 31-269 (right/back); the wall is 270-30.
    check(f"direction {ang}° is on the open side (31-269)",
          31 <= ang <= 269, f"ang={ang}")
    check(f"that corridor is clear (min {mind:.2f} m)", mind > 1.0,
          f"min={mind}")

# All clear -> drives straight ahead (angle 0).
set_ld19([12.0] * 360)
res = rs.tour_goal_direction()
check("all clear -> straight ahead (0°)",
      res is not None and res[0] == 0, f"res={res}")

# LD19 offline -> None.
with rs.LD19_LOCK:
    rs.LD19["online"] = False
check("LD19 offline -> None", rs.tour_goal_direction() is None)
with rs.LD19_LOCK:
    rs.LD19["online"] = True

# --- 2. Goal picking: straight ahead --------------------------------------
print("== tour_pick_goal (straight ahead) ==")
set_ld19(bins)
set_pose(0.0, 0.0, 0.0)
check("picks a goal", rs.tour_pick_goal())
with rs.POLICY_LOCK:
    gx, gy = rs.POLICY["goal_x"], rs.POLICY["goal_y"]
    virt = rs.POLICY["goal_virtual"]
gd = float(np.hypot(gx, gy))
check(f"goal is 0.5-2.5 m out (got {gd:.2f} m)", 0.5 <= gd <= 2.5,
      f"goal=({gx:.2f},{gy:.2f})")
# The goal is STRAIGHT AHEAD (x ~ 0) even though the right side is
# more open — the rover drives straight until a wall, it does not
# steer toward the open corridor.
check("goal is straight ahead, not toward the open side (x ~ 0)",
      abs(gx) < 0.05 and gy > 0,
      f"goal=({gx:.2f},{gy:.2f})")
check("virtual goal is flagged virtual", virt)
# From a different pose the goal moves with the rover.
set_pose(1.0, 0.5, 0.0)
check("repick from a new pose", rs.tour_pick_goal())
with rs.POLICY_LOCK:
    gx2, gy2 = rs.POLICY["goal_x"], rs.POLICY["goal_y"]
check("goal moves with the rover",
      abs(gx2 - gx) > 0.1 or abs(gy2 - gy) > 0.1,
      f"({gx:.2f},{gy:.2f}) -> ({gx2:.2f},{gy2:.2f})")
# The new goal is straight ahead of the new pose too.
check("repicked goal is straight ahead of the new pose",
      abs(gx2 - 1.0) < 0.05 and gy2 > 0.5,
      f"goal=({gx2:.2f},{gy2:.2f})")

# --- 2b. tour_goal_blocked: fixed-point goal validity ----------------------
print("== tour_goal_blocked ==")
# Start fresh, pick a goal straight ahead in open space.
with rs.POLICY_LOCK:
    rs.POLICY["goal_x"] = None
    rs.POLICY["goal_y"] = None
    rs.POLICY["goal_virtual"] = False
set_ld19([12.0] * 360)
set_pose(0.0, 0.0, 0.0)
check("pick a goal straight ahead", rs.tour_pick_goal())
with rs.POLICY_LOCK:
    gx0, gy0 = rs.POLICY["goal_x"], rs.POLICY["goal_y"]
check("goal is straight ahead",
      abs(gx0) < 0.05 and gy0 > 0, f"goal=({gx0:.2f},{gy0:.2f})")
# Open space: nothing between the rover and the goal.
check("open space -> not blocked", not rs.tour_goal_blocked())
# A wall at 1.5 m sits between the rover and the 2.5 m goal.
bins = [12.0] * 360
for b in list(range(345, 360)) + list(range(0, 16)):
    bins[b] = 1.5
set_ld19(bins)
check("wall before the goal -> blocked", rs.tour_goal_blocked())
# A wall BEYOND the goal (3.0 m, goal at 2.5 m) does not block it.
bins = [12.0] * 360
for b in list(range(345, 360)) + list(range(0, 16)):
    bins[b] = 3.0
set_ld19(bins)
check("wall beyond the goal -> not blocked",
      not rs.tour_goal_blocked())
# A wall off to the side (not in the goal's arc) does not block it.
bins = [12.0] * 360
for b in range(80, 101):
    bins[b] = 0.5
set_ld19(bins)
check("wall off to the side -> not blocked",
      not rs.tour_goal_blocked())
# No virtual goal -> not blocked (and no crash).
with rs.POLICY_LOCK:
    rs.POLICY["goal_x"] = None
    rs.POLICY["goal_y"] = None
    rs.POLICY["goal_virtual"] = False
check("no goal -> not blocked", not rs.tour_goal_blocked())

# --- 3. Tour keeps driving ------------------------------------------------
print("== tour mode keeps driving (real ONNX) ==")
check("model loads", rs.policy_load_model(),
      rs.POLICY.get("model_error", ""))

# Record the drive commands.
cmds = []
orig_apply_command = rs.apply_command
orig_apply_drive = rs.apply_drive


def rec_command(cmd, speed_pct):
    cmds.append(("cmd", cmd))
    return orig_apply_command(cmd, speed_pct)


def rec_drive(left, right):
    cmds.append(("drive", round(left, 2), round(right, 2)))
    return orig_apply_drive(left, right)


rs.apply_command = rec_command
rs.apply_drive = rec_drive

# Clear room (env scan with walls) so the policy behaves; pose at
# the origin, heading north.
from rover_env import RoverEnv
_env = RoverEnv()
_env.x, _env.y, _env.heading = 0.0, 0.0, 0.0
_env.gx, _env.gy = 0.0, 1.0
_env.obstacles = []
_scan = _env._scan()
set_ld19([float(_scan[b // 5]) for b in range(360)])
set_pose(0.0, 0.0, 0.0)

# Start the tour the way /tour/on does.
rs.tour_pick_goal()
with rs.POLICY_LOCK:
    goal0 = (rs.POLICY["goal_x"], rs.POLICY["goal_y"])
with rs.TOUR_LOCK:
    rs.TOUR["active"] = True
    rs.TOUR["last_ping"] = time.time()
    rs.TOUR["repick_at"] = time.time() + rs.TOUR_REPICK_S
    rs.TOUR["status"] = "touring…"
with rs.POLICY_LOCK:
    rs.POLICY["tour"] = True
    rs.POLICY["active"] = True
    rs.POLICY["last_ping"] = time.time()
    rs.POLICY["status"] = "touring…"

t = threading.Thread(target=rs.policy_loop, daemon=True)
t.start()
# Let it run ~1.5 s, advancing the pose toward the goal and keeping
# the LD19 + heartbeat fresh. The rover drives straight: the goal
# must stay STRAIGHT AHEAD of it (same x, since it heads north) the
# whole time — it may legitimately repick when it arrives at the
# goal, but the new goal is straight ahead too, never off to a
# side. The loop must stay active.
max_off_axis = 0.0
for _ in range(15):
    time.sleep(0.1)
    with rs.LD19_LOCK:
        rs.LD19["updated"] = time.time()
    with rs.TOUR_LOCK:
        rs.TOUR["last_ping"] = time.time()
    with rs.POLICY_LOCK:
        rs.POLICY["last_ping"] = time.time()
        active = rs.POLICY["active"]
        st = rs.POLICY["status"]
    if not active:
        break
    # Advance the pose toward the current goal (the test plays the rover).
    if cmds and cmds[-1] == ("cmd", "forward"):
        with rs.POSE_LOCK:
            dx = rs.POLICY["goal_x"] - rs.POSE["x"]
            dy = rs.POLICY["goal_y"] - rs.POSE["y"]
            d = float(np.hypot(dx, dy)) or 1.0
            rs.POSE["x"] += 0.03 * dx / d
            rs.POSE["y"] += 0.03 * dy / d
    with rs.POLICY_LOCK:
        with rs.POSE_LOCK:
            off = abs(rs.POLICY["goal_x"] - rs.POSE["x"])
        max_off_axis = max(max_off_axis, off)
with rs.POLICY_LOCK:
    goal1 = (rs.POLICY["goal_x"], rs.POLICY["goal_y"])
drive_cmds = [c for c in cmds if c[0] == "cmd"]
check("tour issues drive commands", len(drive_cmds) > 0,
      f"cmds={cmds[:8]}")
check("tour stays active (never 'arrives')", active, f"status={st}")
check("goal stays STRAIGHT AHEAD while driving (never off to a side)",
      max_off_axis < 0.05,
      f"max off-axis {max_off_axis:.3f} m")
# Now teleport the rover to within 0.5 m of the CURRENT goal
# (simulating arrival) and let the loop run a little: it must
# repick a new goal.
with rs.POSE_LOCK:
    rs.POSE["x"] = goal1[0]
    rs.POSE["y"] = goal1[1] - 0.5
for _ in range(10):
    time.sleep(0.1)
    with rs.LD19_LOCK:
        rs.LD19["updated"] = time.time()
    with rs.TOUR_LOCK:
        rs.TOUR["last_ping"] = time.time()
    with rs.POLICY_LOCK:
        rs.POLICY["last_ping"] = time.time()
with rs.POLICY_LOCK:
    goal2 = (rs.POLICY["goal_x"], rs.POLICY["goal_y"])
check("goal repicked on arrival (point changes)",
      abs(goal2[0] - goal1[0]) > 0.05 or abs(goal2[1] - goal1[1]) > 0.05,
      f"goal stayed {goal1}")
with rs.TOUR_LOCK:
    rs.TOUR["active"] = False
with rs.POLICY_LOCK:
    rs.POLICY["active"] = False
    rs.POLICY["tour"] = False
cmds.clear()
rs.apply_command = orig_apply_command
rs.apply_drive = orig_apply_drive

# --- 4. Live HTTP endpoints ----------------------------------------------
print("== live HTTP endpoints ==")
from http.server import ThreadingHTTPServer
srv = ThreadingHTTPServer(("127.0.0.1", 0), rs.Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()


def post(path):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# The LD19 is fresh and clear, so /tour/on should succeed.
set_ld19([12.0] * 360)
s, b = post("/tour/on")
check("POST /tour/on", s == 200 and "tour on" in b, f"{s} {b}")
s, b = post("/tour/ping")
check("POST /tour/ping returns a status", s == 200 and len(b) > 0,
      f"{s} {b}")
# /policy/on must turn the tour off (and clear the virtual goal).
s, b = post("/policy/goal?x=1.00&y=1.00")
check("set a fixed goal", s == 200 and "goal set" in b, f"{s} {b}")
s, b = post("/policy/on")
check("POST /policy/on", s == 200 and "policy on" in b, f"{s} {b}")
with rs.TOUR_LOCK:
    tour_active = rs.TOUR["active"]
check("policy on turns the tour off", not tour_active)
with rs.POLICY_LOCK:
    tour_flag = rs.POLICY["tour"]
check("policy on clears the tour flag", not tour_flag)
s, b = post("/policy/off")
check("POST /policy/off", s == 200 and "policy off" in b, f"{s} {b}")
# Tour on again, then off.
s, b = post("/tour/on")
check("POST /tour/on (again)", s == 200 and "tour on" in b, f"{s} {b}")
s, b = post("/tour/off")
check("POST /tour/off", s == 200 and "tour off" in b, f"{s} {b}")
with rs.POLICY_LOCK:
    has_goal = rs.POLICY["goal_x"] is not None
check("tour off clears the virtual goal", not has_goal)
# /policy/on with no goal refuses (the virtual goal is gone).
s, b = post("/policy/on")
check("policy on with no goal refuses", s == 400 and "tap the map" in b,
      f"{s} {b}")
srv.shutdown()

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
