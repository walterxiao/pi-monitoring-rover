#!/usr/bin/env python3
"""Verify the heading-sign fix (Nov 23): heading is a compass
bearing, clockwise-positive (0 = north, 90 = east). A left turn
must DECREASE the heading; a right turn must increase it.
1. pose_loop: 1 s left turn -> heading ~240 deg (not ~120).
2. pose_loop: 1 s right turn -> heading increases ~120 deg.
3. pose_loop: forward after turning left from north -> moves west.
4. End-to-end: drive a square (forward + left turns) with the
   real pose_loop + scan_match + map_update_once; the map must
   show clean walls at the true positions, not a blob.
"""
import importlib.util
import math
import sys
import threading
import time

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "rover", "/home/hatch/workspace/pi-monitoring-rover/rover_server.py")
rover = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rover)

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def pose():
    with rover.POSE_LOCK:
        return (rover.POSE["x"], rover.POSE["y"], rover.POSE["heading"])


def set_throttles(tl, tr):
    with rover.LOCK:
        rover.STATE["throttle_l"] = tl
        rover.STATE["throttle_r"] = tr
        rover.STATE["moving"] = True
        rover.STATE["last_cmd_time"] = time.time() + 100


def stop_pose():
    with rover.LOCK:
        rover.STATE["moving"] = False
        rover.STATE["throttle_l"] = 0.0
        rover.STATE["throttle_r"] = 0.0


# --- 1-3. pose_loop heading sign ------------------------------------------
t = threading.Thread(target=rover.pose_loop, daemon=True)
t.start()
time.sleep(0.2)
rover.map_clear()
x0, y0, h0 = pose()
check("pose starts at origin, heading 0", h0 == 0.0, f"got {pose()}")

# Left turn 1.0 s: heading must DECREASE by ~120 deg (0 -> ~240).
set_throttles(1.0, -1.0)
time.sleep(1.0)
stop_pose()
time.sleep(0.2)
x1, y1, h1 = pose()
check("left turn 1s -> heading decreases ~120 deg (to ~240)",
      225 < h1 < 255, f"got heading={h1:.1f} (240 = west-ish, 120 = WRONG way)")
check("left turn -> position ~unchanged",
      abs(x1 - x0) < 0.05 and abs(y1 - y0) < 0.05,
      f"got ({x1:.2f}, {y1:.2f})")

# Right turn 1.0 s: heading must INCREASE by ~120 deg (240 -> ~0).
set_throttles(-1.0, 1.0)
time.sleep(1.0)
stop_pose()
time.sleep(0.2)
x2, y2, h2 = pose()
check("right turn 1s -> heading increases ~120 deg (back to ~0)",
      h2 < 15 or h2 > 345, f"got heading={h2:.1f}")

# Turn left 0.75 s (heading -> ~270, facing west), then drive
# forward 1 s: the rover must move WEST (x decreases), not east.
set_throttles(1.0, -1.0)
time.sleep(0.75)
stop_pose()
time.sleep(0.2)
x3, y3, h3 = pose()
check("left turn 0.75s -> heading ~270 (facing west)",
      255 < h3 < 285, f"got heading={h3:.1f}")
set_throttles(-1.0, -1.0)
time.sleep(1.0)
stop_pose()
time.sleep(0.2)
x4, y4, h4 = pose()
check("forward while facing west -> moves WEST (x decreases)",
      x4 < x3 - 0.20 and abs(y4 - y3) < 0.05,
      f"moved ({x3:.2f},{y3:.2f}) -> ({x4:.2f},{y4:.2f})")

# --- 4. End-to-end square drive + map --------------------------------------
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
                tt = wall_d / dx
                if tt <= 0:
                    continue
                if abs(ry + tt * dy) <= 1.5:
                    best = tt if best is None else min(best, tt)
            else:
                if abs(dy) < 1e-9:
                    continue
                tt = wall_d / dy
                if tt <= 0:
                    continue
                if abs(rx + tt * dx) <= 2.0:
                    best = tt if best is None else min(best, tt)
        bins.append(best)
    return bins


rover.map_clear()
np.random.seed(7)
# True pose, tracked with correct compass physics (heading
# clockwise-positive; left turn decreases it). Starts at (0, 0)
# to match map_clear's pose reset.
tx, ty, th = 0.0, 0.0, 0.0
# Seed the map at the true start pose.
rover.map_update_once(room_scan(tx, ty, th), tx, ty, th)
# Drive a square: forward 1 s, left turn 0.75 s (90 deg), repeat.
# Each leg: 0.3 m forward; each turn: 90 deg left.
max_pose_err = 0.0
for leg in range(4):
    # Forward 1 s (true: 0.3 m along heading).
    set_throttles(-1.0, -1.0)
    time.sleep(1.0)
    stop_pose()
    time.sleep(0.15)
    thr = math.radians(th)
    tx += 0.3 * math.sin(thr)
    ty += 0.3 * math.cos(thr)
    px, py, ph = pose()
    err = math.hypot(px - tx, py - ty)
    herr = abs((ph - th + 180) % 360 - 180)
    max_pose_err = max(max_pose_err, err)
    # Map update with the pose_loop pose (scan-matched).
    bins = room_scan(tx, ty, th)
    cx, cy, ch = rover.scan_match(bins, px, py, ph)
    rover.map_update_once(bins, cx, cy, ch)
    # Left turn 0.75 s (true: heading -90 deg).
    set_throttles(1.0, -1.0)
    time.sleep(0.75)
    stop_pose()
    time.sleep(0.15)
    th = (th - 90.0) % 360.0
    px, py, ph = pose()
    herr = abs((ph - th + 180) % 360 - 180)
    max_pose_err = max(max_pose_err, math.hypot(px - tx, py - ty))
    bins = room_scan(tx, ty, th)
    cx, cy, ch = rover.scan_match(bins, px, py, ph)
    rover.map_update_once(bins, cx, cy, ch)

check("square drive: pose_loop tracks true pose",
      max_pose_err < 0.08, f"max pose err={max_pose_err:.3f} m")
st = rover.map_stats()
check("square drive: map has a sane wall count",
      200 < st["occupied"] < 3000,
      f"occupied={st['occupied']} (blob would be >>3000)")
grid = rover.MAP["grid"]
# The north wall (y=+1.5) must be a clean line at the right row,
# and the area 0.8 m south of it (where a smeared wall would be)
# must be mostly clear.
north_row = rover.MAP_CENTER - int(round(1.5 / rover.MAP_CELL_M))
wall_band = int((grid[north_row - 1:north_row + 2, :] == 2).sum())
smear_row = north_row + int(round(0.8 / rover.MAP_CELL_M))
smear_band = int((grid[smear_row - 1:smear_row + 2, :] == 2).sum())
check("square drive: north wall is a clean line, not smeared",
      wall_band > 20 and smear_band < wall_band / 2,
      f"wall band={wall_band}, smeared band={smear_band}")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
