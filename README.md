# Pi Monitoring Rover

A drive-around-the-house camera rover built from a Raspberry Pi 5 +
ArduCam camera. Phone-controlled, no soldering, no autonomy (v1).
Build time: **60–90 minutes**, screwdriver work only.

Two separate power systems: a USB-C power bank runs the Pi 5, and
4×AA batteries run the motors through an Adafruit DC & Stepper Motor
HAT — so motor noise can never reboot the Pi mid-drive.

## Files

- `rover_server.py` — the drive server. Run it on the Pi, then open
  `http://<pi-address>:8001` on your phone: big hold-to-drive buttons.
  A watchdog stops the motors within ~1.5 seconds if commands stop
  arriving, so the rover can never run away on its own.
- `test_motors.py` — spins one motor at a time (wheels in the air) so
  you can check wiring before driving.
- `BUILD_GUIDE.md` — the full step-by-step build guide.
- `PARTS_LIST.md` — chassis, motor HAT, batteries, and what you
  already own.

## Motor map

M1 = front-left, M2 = rear-left, M3 = front-right, M4 = rear-right.
If a wheel spins backwards, power off and swap that motor's two wires
in its HAT terminal block.
