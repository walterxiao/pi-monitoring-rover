#!/usr/bin/env python3
"""Verify the learned-policy (ONNX) mode in rover_server.py.

1. Observation parity: policy_observation() must match RoverEnv._obs()
   for the same pose/goal/scan (the math the policy was trained on).
2. Policy loop with the REAL trained ONNX model (the 50k-step smoke
   run): drives forward to a clear goal, turns when the safety
   override fires, stops at the goal, stops when the LD19 goes stale.
3. Live HTTP endpoints: /policy/goal, /policy/on, /policy/ping,
   /policy/off, /map.json carries the goal.
"""
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/home/hatch/workspace/goals/mobile-pi-monitoring-rover/files/rl_rover")
sys.path.insert(0, "/home/hatch/workspace/pi-monitoring-rover")

import numpy as np

import rover_env
rover_env.LIDAR_NOISE_M = 0.0
rover_env.LIDAR_DROPOUT_P = 0.0
from rover_env import RoverEnv

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


# --- 1. Observation parity -----------------------------------------------
print("== observation parity (env vs Pi) ==")
for seed in (42, 7, 123):
    env = RoverEnv()
    env.reset(seed=seed)
    scan = env._scan()  # 72 clean rays, 5 deg apart, ray 0 = heading
    bins = [float(scan[b // 5]) for b in range(360)]
    with rs.LD19_LOCK:
        rs.LD19["online"] = True
        rs.LD19["updated"] = time.time()
        rs.LD19["bins"] = bins
    with rs.POSE_LOCK:
        rs.POSE["x"] = env.x
        rs.POSE["y"] = env.y
        rs.POSE["heading"] = env.heading
    with rs.POLICY_LOCK:
        rs.POLICY["goal_x"] = env.gx
        rs.POLICY["goal_y"] = env.gy
    obs_pi = rs.policy_observation()
    obs_env = env._obs()
    d = float(np.max(np.abs(obs_pi - obs_env)))
    check(f"seed {seed} obs matches env (max diff {d:.2e})", d < 1e-6,
          f"diff={d}")

# Stale LD19 -> None; no goal -> None.
with rs.LD19_LOCK:
    rs.LD19["updated"] = time.time() - 5.0
check("stale LD19 -> None", rs.policy_observation() is None)
with rs.LD19_LOCK:
    rs.LD19["updated"] = time.time()
with rs.POLICY_LOCK:
    rs.POLICY["goal_x"] = None
    rs.POLICY["goal_y"] = None
check("no goal -> None", rs.policy_observation() is None)

# --- 2. Policy loop with the real ONNX model ------------------------------
print("== policy loop (real ONNX, mocked sensors) ==")
SMOKE_ONNX = ("/home/hatch/workspace/goals/mobile-pi-monitoring-rover/"
              "files/rl_rover/runs/smoke/rover_policy.onnx")
rs.POLICY_ONNX_PATH = SMOKE_ONNX
check("model loads", rs.policy_load_model(),
      rs.POLICY.get("model_error", ""))
# Sanity: the model returns a valid action for a real observation.
env = RoverEnv()
env.reset(seed=42)
scan = env._scan()
bins = [float(scan[b // 5]) for b in range(360)]
with rs.LD19_LOCK:
    rs.LD19["online"] = True
    rs.LD19["updated"] = time.time()
    rs.LD19["bins"] = bins
with rs.POSE_LOCK:
    rs.POSE["x"] = env.x
    rs.POSE["y"] = env.y
    rs.POSE["heading"] = env.heading
with rs.POLICY_LOCK:
    rs.POLICY["goal_x"] = env.gx
    rs.POLICY["goal_y"] = env.gy
obs = rs.policy_observation()
with rs.POLICY_LOCK:
    sess = rs.POLICY["model"]
act = int(sess.run(None, {"observation": obs[None, :]})[0][0])
check(f"real ONNX returns an action (got {act})", act in (0, 1, 2, 3))

# Record the drive commands the policy issues.
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

# Fake the TF-Luna as clear (far away) so only the LD19 matters.
rs.lidar_reading = lambda: (8.0, 100)

# Scenario A: clear room, goal 1.0 m straight ahead (in-distribution:
# the training goals keep a 0.4 m margin from the walls). The policy
# should drive forward (and the pose, advanced by the test as if
# the rover moved, should close on the goal). Then teleport the
# pose onto the goal and confirm the loop stops with "goal reached".
with rs.POSE_LOCK:
    rs.POSE["x"] = 0.0
    rs.POSE["y"] = 0.0
    rs.POSE["heading"] = 0.0
with rs.POLICY_LOCK:
    rs.POLICY["goal_x"] = 0.0
    rs.POLICY["goal_y"] = 1.0
    rs.POLICY["active"] = True
    rs.POLICY["last_ping"] = time.time()
    rs.POLICY["status"] = "test"
# LD19: the env's real scan for this pose (walls included) — an
# all-clear scan is out of distribution (training always has walls),
# so the policy needs the real room geometry to behave.
_env = RoverEnv()
_env.x, _env.y, _env.heading = 0.0, 0.0, 0.0
_env.gx, _env.gy = 0.0, 1.0
_env.obstacles = []
_scan = _env._scan()
with rs.LD19_LOCK:
    rs.LD19["online"] = True
    rs.LD19["updated"] = time.time()
    rs.LD19["bins"] = [float(_scan[b // 5]) for b in range(360)]

t = threading.Thread(target=rs.policy_loop, daemon=True)
t.start()
# Drive the pose forward by hand (the test plays the rover): each
# 0.1 s, if the last command was forward, move 0.03 m north.
# Keep the LD19 fresh so the loop doesn't stop.
for _ in range(30):
    time.sleep(0.1)
    with rs.LD19_LOCK:
        rs.LD19["updated"] = time.time()
    with rs.POLICY_LOCK:
        rs.POLICY["last_ping"] = time.time()
    if cmds and cmds[-1] == ("cmd", "forward"):
        with rs.POSE_LOCK:
            rs.POSE["y"] += 0.03
fwd_cmds = [c for c in cmds if c == ("cmd", "forward")]
with rs.POSE_LOCK:
    y_now = rs.POSE["y"]
check("clear room: policy drives forward", len(fwd_cmds) > 0,
      f"cmds={cmds[:8]}")
check("clear room: pose closes on the goal", y_now > 0.3,
      f"y={y_now:.2f}")
# Teleport onto the goal (0.15 m away) — the loop must stop.
with rs.POSE_LOCK:
    rs.POSE["y"] = 0.85
reached = False
st = ""
for _ in range(20):
    time.sleep(0.1)
    with rs.LD19_LOCK:
        rs.LD19["updated"] = time.time()
    with rs.POLICY_LOCK:
        rs.POLICY["last_ping"] = time.time()
        active = rs.POLICY["active"]
        st = rs.POLICY["status"]
    if not active:
        reached = "goal reached" in st
        break
check("at the goal: stops with 'goal reached'", reached, f"status={st}")
cmds.clear()

# Scenario B: wall dead ahead at 0.30 m (under the 0.50 m min).
# The safety override must turn, never drive forward.
with rs.POSE_LOCK:
    rs.POSE["x"] = 0.0
    rs.POSE["y"] = 0.0
    rs.POSE["heading"] = 0.0
with rs.POLICY_LOCK:
    rs.POLICY["goal_x"] = 0.0
    rs.POLICY["goal_y"] = 1.0
    rs.POLICY["active"] = True
    rs.POLICY["last_ping"] = time.time()
bins = [12.0] * 360
for b in range(345, 360):
    bins[b] = 0.30
for b in range(0, 16):
    bins[b] = 0.30
with rs.LD19_LOCK:
    rs.LD19["online"] = True
    rs.LD19["updated"] = time.time()
    rs.LD19["bins"] = bins
time.sleep(0.6)
with rs.LD19_LOCK:
    rs.LD19["updated"] = time.time()
with rs.POLICY_LOCK:
    rs.POLICY["last_ping"] = time.time()
    rs.POLICY["active"] = False
fwd = [c for c in cmds if c == ("cmd", "forward")]
turns = [c for c in cmds if c in (("cmd", "left"), ("cmd", "right"))]
check("blocked ahead: never drives forward", len(fwd) == 0,
      f"cmds={cmds[:8]}")
check("blocked ahead: turns instead", len(turns) > 0, f"cmds={cmds[:8]}")
cmds.clear()

# Scenario C: LD19 goes stale mid-run -> stops, does not drive blind.
with rs.POLICY_LOCK:
    rs.POLICY["goal_x"] = 0.0
    rs.POLICY["goal_y"] = 1.0
    rs.POLICY["active"] = True
    rs.POLICY["last_ping"] = time.time()
with rs.LD19_LOCK:
    rs.LD19["bins"] = [12.0] * 360
    rs.LD19["updated"] = time.time() - 5.0  # stale
time.sleep(0.4)
with rs.POLICY_LOCK:
    rs.POLICY["active"] = False
    st = rs.POLICY["status"]
stops = [c for c in cmds if c == ("drive", 0, 0) or c == ("cmd", "stop")]
check("stale LD19: stops", len(stops) > 0 or "offline" in st,
      f"status={st} cmds={cmds[:5]}")
cmds.clear()
rs.apply_command = orig_apply_command
rs.apply_drive = orig_apply_drive

# --- 3. Live HTTP endpoints ----------------------------------------------
print("== live HTTP endpoints ==")
# Pretend the motors are present (this VM has no HAT); the real Pi does.
rs.MOTORS_OK = True
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


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=5) as r:
        return r.status, r.read().decode()


import json
s, b = post("/policy/goal?x=1.00&y=2.00")
check("POST /policy/goal", s == 200 and "goal set" in b, f"{s} {b}")
s, b = get("/map.json")
d = json.loads(b)
check("/map.json carries the goal", d.get("goal_x") == 1.0
      and d.get("goal_y") == 2.0, b[:200])
# /policy/on needs the model; the smoke ONNX is pointed at already.
s, b = post("/policy/on")
check("POST /policy/on", s == 200 and "policy on" in b, f"{s} {b}")
s, b = post("/policy/ping")
check("POST /policy/ping returns a status", s == 200 and len(b) > 0,
      f"{s} {b}")
s, b = post("/policy/off")
check("POST /policy/off", s == 200 and "policy off" in b, f"{s} {b}")
# No goal -> /policy/on refuses.
with rs.POLICY_LOCK:
    rs.POLICY["goal_x"] = None
    rs.POLICY["goal_y"] = None
s, b = post("/policy/on")
check("POST /policy/on without a goal refuses", s == 400
      and "tap the map" in b, f"{s} {b}")
srv.shutdown()

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
