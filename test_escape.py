#!/usr/bin/env python3
"""Verify minimum-angle escape (Nov 23): the rover picks the
smaller turn (left or right) to open space instead of always
turning left and circling a corner.
1. Corner (wall ahead + left, open right) -> turns RIGHT.
2. Mirror corner (wall ahead + right, open left) -> turns LEFT.
3. Surrounded (no clear direction) -> None.
4. LD19 offline -> None.
5. avoid_loop in a corner turns RIGHT (not left) and drives out.
"""
import importlib.util
import math
import sys
import threading
import time

SPEC = importlib.util.spec_from_file_location(
    "rover", "/home/hatch/workspace/pi-monitoring-rover/rover_server.py")
rover = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rover)

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# Room: walls at x = +/-2 m, y = +/-1.5 m.
def room_scan(rx, ry, heading_deg):
    bins = []
    for b in range(360):
        a = math.radians((heading_deg + b) % 360)
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
        bins.append(best)
    return bins


def set_ld19_scan(bins):
    with rover.LD19_LOCK:
        rover.LD19["online"] = True
        rover.LD19["bins"] = list(bins)
        rover.LD19["updated"] = time.time() + 100  # stay fresh


STOP_M = 0.5

# --- 1. Corner: wall ahead + left, open right -> RIGHT --------------------
# Rover at (-1.5, 1.1) facing north: north wall 0.4 m ahead, west
# wall 0.5 m to the left, open to the east.
bins = room_scan(-1.5, 1.1, 0.0)
set_ld19_scan(bins)
esc = rover.ld19_escape_direction(STOP_M)
check("corner (wall ahead+left, open right) -> turns RIGHT",
      esc is not None and esc[0] == "right",
      f"got {esc}")
if esc is not None:
    check("corner escape angle is sane (<= 120 deg)",
          0 < esc[1] <= 120, f"got {esc[1]} deg")

# --- 2. Mirror corner: wall ahead + right, open left -> LEFT ---------------
# Rover at (1.5, 1.1) facing north: north wall ahead, east wall to
# the right, open to the west.
bins = room_scan(1.5, 1.1, 0.0)
set_ld19_scan(bins)
esc = rover.ld19_escape_direction(STOP_M)
check("mirror corner (wall ahead+right, open left) -> turns LEFT",
      esc is not None and esc[0] == "left",
      f"got {esc}")

# --- 3. Surrounded -> None --------------------------------------------------
# Tiny box: rover at origin, walls 0.4 m away in every direction.
def box_scan():
    return [0.4] * 360
set_ld19_scan(box_scan())
esc = rover.ld19_escape_direction(STOP_M)
check("surrounded (no clear direction) -> None", esc is None,
      f"got {esc}")

# --- 4. LD19 offline -> None ----------------------------------------------
with rover.LD19_LOCK:
    rover.LD19["online"] = False
esc = rover.ld19_escape_direction(STOP_M)
check("LD19 offline -> None", esc is None, f"got {esc}")

# --- 5. avoid_loop in a corner turns RIGHT and drives out ------------------
# Rover at (-1.5, 1.1) facing north, in the northwest corner.
# The north wall blocks the front; the west wall blocks the left;
# the east is open. Run the real avoid_loop with a simulated scan
# that follows the rover's pose, and check it turns right (not
# left) and then drives forward away from the corner.
rover.map_clear()
# Mock the drive calls to record them (and let the real ones set
# STATE so pose_loop integrates).
calls = []
orig_drive = rover.apply_drive
orig_cmd = rover.apply_command
def rec_drive(l, r):
    calls.append(("drive", l, r))
    return orig_drive(l, r)
def rec_cmd(c, s=100):
    calls.append(("cmd", c, s))
    return orig_cmd(c, s)
rover.apply_drive = rec_drive
rover.apply_command = rec_cmd
rover.avoid_set_status = lambda s: None
# Start pose_loop so the pose integrates from the throttles.
t_pose = threading.Thread(target=rover.pose_loop, daemon=True)
t_pose.start()
time.sleep(0.2)
# True pose for the scan simulation (starts at the corner).
true_x, true_y, true_h = -1.5, 1.1, 0.0
# Set the pose to the corner (map_clear put it at 0,0; override).
with rover.POSE_LOCK:
    rover.POSE["x"], rover.POSE["y"], rover.POSE["heading"] = true_x, true_y, true_h
with rover.LIDAR_LOCK:
    rover.HEADING["deg"] = true_h
# TF-Luna: blocked (0.4 m) so the rover is blocked even if the
# LD19 front arc were clear; refresh it in the scan thread.
stop_scan = threading.Event()
def scan_thread():
    while not stop_scan.is_set():
        with rover.POSE_LOCK:
            px, py, ph = rover.POSE["x"], rover.POSE["y"], rover.POSE["heading"]
        bins = room_scan(px, py, ph)
        set_ld19_scan(bins)
        # TF-Luna = the forward beam: same as the LD19's bin 0
        # (straight ahead), so it updates as the rover turns.
        with rover.LIDAR_LOCK:
            rover.LIDAR["online"] = True
            rover.LIDAR["dist_m"] = bins[0]
            rover.LIDAR["strength"] = 100
            rover.LIDAR["updated"] = time.time() + 100
        time.sleep(0.05)
st = threading.Thread(target=scan_thread, daemon=True)
st.start()
with rover.AVOID_LOCK:
    rover.AVOID["active"] = True
    rover.AVOID["last_ping"] = time.time() + 100
    rover.AVOID["min_m"] = STOP_M
t_avoid = threading.Thread(target=rover.avoid_loop, daemon=True)
t_avoid.start()
time.sleep(3.0)
with rover.AVOID_LOCK:
    rover.AVOID["active"] = False
stop_scan.set()
time.sleep(0.3)
rover.apply_drive = orig_drive
rover.apply_command = orig_cmd
# Did it turn right (not left)?
right_cmds = [c for c in calls if c[0] == "cmd" and c[1] == "right"]
left_cmds = [c for c in calls if c[0] == "cmd" and c[1] == "left"]
fwd = [c for c in calls if c[0] == "drive" and c[1] < 0 and c[2] < 0]
check("avoid in corner turns RIGHT (not left)",
      len(right_cmds) > 0 and len(left_cmds) == 0,
      f"right={len(right_cmds)}, left={len(left_cmds)}")
check("avoid in corner then drives forward",
      len(fwd) > 0, f"forward drive calls={len(fwd)}")
with rover.POSE_LOCK:
    ex, ey, eh = rover.POSE["x"], rover.POSE["y"], rover.POSE["heading"]
check("avoid in corner moves the rover away (east / +x)",
      ex > -1.5 + 0.1, f"pose=({ex:.2f}, {ey:.2f}, {eh:.0f})")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
