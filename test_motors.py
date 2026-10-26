#!/usr/bin/env python3
"""Rover motor test — spins ONE motor at a time so you can check wiring.

Run this with the rover propped up, ALL WHEELS IN THE AIR:
    python3 test_motors.py

Motor map (matches BUILD_GUIDE.md):
    M1 = front-left   M2 = rear-left   M3 = front-right   M4 = rear-right

If a wheel spins backwards (top of wheel moving toward the REAR of the
car when it should go forward), power off and swap that motor's two
wires in its terminal block on the HAT.
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
MOTORS = [
    ("M1", kit.motor1, "front-left"),
    ("M2", kit.motor2, "rear-left"),
    ("M3", kit.motor3, "front-right"),
    ("M4", kit.motor4, "rear-right"),
]


def stop_all():
    for _, motor, _ in MOTORS:
        motor.throttle = None


def main():
    print(__doc__)
    input("Wheels in the air and battery box plugged in? Press Enter... ")
    try:
        for label, motor, where in MOTORS:
            print(f"\n>>> {label} ({where}) spinning SLOWLY for 2 seconds...")
            motor.throttle = 0.4
            time.sleep(2)
            motor.throttle = None
            answer = input(
                f"    Did the {where} wheel spin FORWARD "
                "(top of wheel toward the front)? [y/n] "
            ).strip().lower()
            if answer != "y":
                print(f"    -> Noted: swap the two {label} wires in the "
                      "terminal block (power off first), then re-run.")
            else:
                print("    -> Good.")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        stop_all()
    print("\nDone. All motors stopped.")


if __name__ == "__main__":
    main()
