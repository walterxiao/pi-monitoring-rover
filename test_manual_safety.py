#!/usr/bin/env python3
"""Verify manual safety (Nov 24): D-pad driving slows down as an
obstacle approaches and stops before hitting it.
1. Far obstacle (>= 1.0 m) -> full speed (factor 1.0).
2. Mid-range obstacle (0.65 m) -> half speed (factor ~0.5).
3. Close obstacle (<= 0.30 m) -> stop (factor 0.0).
4. Clear (no obstacle) -> full speed.
5. Sensors offline -> full speed (drives unprotected, as before).
6. Reverse: rear obstacle close -> stop; rear clear -> full speed.
7. Turns are unscaled (factor 1.0 even with a wall ahead).
8. End-to-end: manual_safety_loop caps a forward drive as a wall
   approaches and stops it at 0.30 m.
"""
import importlib.util
import math
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


def set_sensors(fwd_ld19=None, tf=None, rear_ld19=None):
    """Set the LD19 bins and TF-Luna reading. fwd_ld19/rear_ld19
    are distances (m) in the forward/rear arcs (None = clear);
    tf is the TF-Luna distance (None = offline/clear)."""
    with rover.LD19_LOCK:
        rover.LD19["online"] = True
        rover.LD19["bins"] = [None] * 360
        if fwd_ld19 is not None:
            for a in range(-30, 31):
                rover.LD19["bins"][a % 360] = fwd_ld19
        if rear_ld19 is not None:
            for a in range(135, 226):
                rover.LD19["bins"][a % 360] = rear_ld19
        rover.LD19["updated"] = time.time() + 100
    with rover.LIDAR_LOCK:
        if tf is None:
            rover.LIDAR["online"] = False
        else:
            rover.LIDAR["online"] = True
            rover.LIDAR["dist_m"] = tf
            rover.LIDAR["strength"] = 100
            rover.LIDAR["updated"] = time.time() + 100


# --- 1. Far obstacle -> full speed ---------------------------------------
set_sensors(fwd_ld19=1.5, tf=1.5)
check("far obstacle (1.5 m) -> full speed",
      rover.manual_speed_factor("forward") == 1.0,
      f"got {rover.manual_speed_factor('forward')}")

# --- 2. Mid-range -> half speed -------------------------------------------
set_sensors(fwd_ld19=0.65, tf=0.65)
f = rover.manual_speed_factor("forward")
check("mid-range obstacle (0.65 m) -> ~half speed",
      0.4 < f < 0.6, f"got {f:.2f}")

# --- 3. Close obstacle -> stop ---------------------------------------------
set_sensors(fwd_ld19=0.25, tf=0.25)
check("close obstacle (0.25 m) -> stop",
      rover.manual_speed_factor("forward") == 0.0,
      f"got {rover.manual_speed_factor('forward')}")

# --- 3b. At exactly the stop distance -> stop ------------------------------
set_sensors(fwd_ld19=0.30, tf=0.30)
check("at stop distance (0.30 m) -> stop",
      rover.manual_speed_factor("forward") == 0.0,
      f"got {rover.manual_speed_factor('forward')}")

# --- 3c. TF-Luna sees it even if the LD19 is clear ------------------------
set_sensors(fwd_ld19=None, tf=0.25)
check("TF-Luna close, LD19 clear -> stop",
      rover.manual_speed_factor("forward") == 0.0,
      f"got {rover.manual_speed_factor('forward')}")

# --- 4. Clear -> full speed -------------------------------------------------
set_sensors(fwd_ld19=None, tf=None)
# TF-Luna offline + LD19 clear (bins all None) -> inf -> full speed.
check("clear (no obstacle) -> full speed",
      rover.manual_speed_factor("forward") == 1.0,
      f"got {rover.manual_speed_factor('forward')}")

# --- 5. Sensors offline -> full speed --------------------------------------
with rover.LD19_LOCK:
    rover.LD19["online"] = False
with rover.LIDAR_LOCK:
    rover.LIDAR["online"] = False
check("sensors offline -> full speed (unprotected, as before)",
      rover.manual_speed_factor("forward") == 1.0,
      f"got {rover.manual_speed_factor('forward')}")

# --- 6. Reverse -------------------------------------------------------------
set_sensors(rear_ld19=0.25)
check("reverse, rear obstacle close (0.25 m) -> stop",
      rover.manual_speed_factor("back") == 0.0,
      f"got {rover.manual_speed_factor('back')}")
set_sensors(rear_ld19=None)
check("reverse, rear clear -> full speed",
      rover.manual_speed_factor("back") == 1.0,
      f"got {rover.manual_speed_factor('back')}")

# --- 7. Turns unscaled -------------------------------------------------------
set_sensors(fwd_ld19=0.25, tf=0.25)
check("turn left with wall ahead -> unscaled (1.0)",
      rover.manual_speed_factor("left") == 1.0)
check("turn right with wall ahead -> unscaled (1.0)",
      rover.manual_speed_factor("right") == 1.0)
check("stop with wall ahead -> unscaled (1.0)",
      rover.manual_speed_factor("stop") == 1.0)

# --- 8. End-to-end: safety loop caps a forward drive ------------------------
# Rover drives forward toward a wall. The wall starts 1.2 m away
# and the rover approaches at 0.3 m/s. The safety loop should
# slow it down and stop it before it gets within 0.30 m.
rover.MOTORS_OK = True
orig_set_motors = rover.set_motors
orig_stop_all = rover.stop_all
applied = []
def rec_set_motors(l, r):
    # Like the real set_motors, this must NOT take LOCK: the
    # caller (apply_command / the safety loop) already holds it.
    applied.append((l, r))
rover.set_motors = rec_set_motors
def rec_stop_all():
    # Same: no LOCK here (the caller holds it).
    applied.append((0.0, 0.0))
rover.stop_all = rec_stop_all
# Start the safety loop.
t_safe = threading.Thread(target=rover.manual_safety_loop, daemon=True)
t_safe.start()
time.sleep(0.2)
# Simulate driving forward: set throttles to full forward and
# update the sensor as the rover "approaches" the wall. Refresh
# last_cmd_time each step (as the page's 500 ms re-send does) so
# the safety loop stays active.
wall_dist = 1.2
rover.apply_command("forward", 100)
for step in range(100):
    # Rover moves 0.3 m/s * 0.1 s = 0.03 m per step (at full speed;
    # the safety loop slows it, so use the actual applied speed).
    with rover.LOCK:
        cur = abs(rover.STATE["throttle_l"])
        rover.STATE["last_cmd_time"] = time.time()
    wall_dist -= 0.3 * cur * 0.1
    if wall_dist < 0.05:
        wall_dist = 0.05
    set_sensors(fwd_ld19=wall_dist, tf=wall_dist)
    time.sleep(0.1)
    with rover.LOCK:
        moving = rover.STATE["moving"]
        tl = rover.STATE["throttle_l"]
    if not moving:
        break
with rover.LOCK:
    final_tl = rover.STATE["throttle_l"]
    final_moving = rover.STATE["moving"]
rover.set_motors = orig_set_motors
rover.stop_all = orig_stop_all
check("safety loop stops the rover before it hits the wall",
      not final_moving and final_tl == 0.0,
      f"moving={final_moving}, throttle={final_tl}")
check("safety loop stopped it at >= 0.30 m from the wall",
      wall_dist >= 0.29,
      f"final wall_dist={wall_dist:.2f} m")
# It should have slowed down (not just run at full speed then
# stopped): check that some applied throttle was < full speed.
slowed = [a for a in applied if 0 < abs(a[0]) < 1.0]
check("safety loop slowed the rover down before stopping",
      len(slowed) > 0, f"applied throttles: {applied[:10]}")

print()
if fails:
    print(f"FAILURES: {fails}")
else:
    print("FAILURES: none")
