#!/usr/bin/env python3
"""Find out which wheel each HAT motor channel actually drives.

The drive code assumes M1/M2 are the LEFT wheels and M3/M4 are the
RIGHT wheels, but the rover's turns don't work — the front wheels
spin one way and the rear wheels the other, so they fight each other
and the rover goes nowhere. That means the motors are plugged in
differently than the code assumes (probably front pair / rear pair).

Run this with the rover propped up, ALL WHEELS IN THE AIR, battery
box plugged in, and the drive server STOPPED first:
    sudo systemctl stop rover-server
    python3 test_motor_map.py
    sudo systemctl start rover-server

It spins M1, then M2, then M3, then M4, one at a time, 3 seconds
each. Watch which wheel spins each time and write it down, then
tell Luffy: M1 = ?, M2 = ?, M3 = ?, M4 = ?
(front-left / front-right / rear-left / rear-right)
"""
import time

try:
    from adafruit_motorkit import MotorKit
except ImportError:
    raise SystemExit(
        "Motor library not found. Install it with:\n"
        "  sudo pip3 install --break-system-packages "
        "adafruit-circuitpython-motorkit"
    )

kit = MotorKit()


def stop_all():
    for i in range(1, 5):
        getattr(kit, f"motor{i}").throttle = None


def main():
    print("Spinning each motor one at a time. Watch which wheel moves.")
    print("Front = the camera end. Wheels in the air, please!\n")
    try:
        for i in range(1, 5):
            motor = getattr(kit, f"motor{i}")
            print(f">>> M{i} spinning for 3 seconds... which wheel moves?",
                  flush=True)
            motor.throttle = 0.5
            time.sleep(3)
            motor.throttle = None
            print(f"    M{i} done. (front-left / front-right / "
                  "rear-left / rear-right?)", flush=True)
            time.sleep(1.5)
    finally:
        stop_all()
    print("\nDone. Tell Luffy: M1 = ?, M2 = ?, M3 = ?, M4 = ?")


if __name__ == "__main__":
    main()
