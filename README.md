# Pi Monitoring Rover

A drive-around-the-house camera rover built from a Raspberry Pi 5 +
ArduCam camera. Phone-controlled over Wi-Fi, with a learned
reinforcement-learning driving policy, a live floor-plan map, and
dual-LiDAR safety. No soldering. Build time: **60–90 minutes**,
screwdriver work only.

Two separate power systems: a USB-C power bank runs the Pi 5, and
4×AA batteries run the motors through an Adafruit DC & Stepper Motor
HAT — so motor noise can never reboot the Pi mid-drive.

## Drive page

Run the server on the Pi, then open `http://<pi-address>:8001` on your
phone (same Wi-Fi). The camera video fills the screen.

- **D-pad** — circular, lower-left: ▲ forward, ▼ reverse, ◀ ▶ spin in
  place. Hold to drive, let go to stop. Keyboard arrow keys drive the
  same way on a computer. The center button (↻ 180°) spins the rover a
  half-turn in place; any D-pad button cancels it mid-turn.
- **Line Follow** — drives itself along a blue tape line on the floor.
- **Tour** — the rover explores on its own (see below).
- **Go to goal** — tap the map to set a destination; the rover drives
  there and stops within 0.25 m.
- **Radar** — 300 px circular view of the LD19's 360° scan, with the
  TF-Luna's single forward beam as a dot.
- **Map** — a live floor plan built from the LD19 scans (see below).

A watchdog stops the motors within ~1.5 seconds if commands stop
arriving, so the rover can never run away on its own.

## Sensors

- **LD19 360° LiDAR** — the rover's eyes. Mounted on a 3D-printed
  gantry (`ld19_gantry.py`) above the Pi. Draws the radar and builds
  the map. Its 0° is rotated 90° in software (`LD19_ANGLE_OFFSET`)
  so the radar's front matches the rover's front.
- **TF-Luna single-point LiDAR** — watches the low front, where the
  LD19's plane can miss low obstacles.
- Both LiDARs **auto-detect their USB port by data signature**, so
  they keep working after a reboot even if the adapters swap ports.

## Tour mode (autonomous explore)

Tour is the explore mode. A reinforcement-learning policy (PPO,
trained in simulation, run on the Pi as ONNX) drives the rover toward
a virtual goal placed **straight ahead** — 80% of the forward
clearance, capped at 2.5 m. The goal is a fixed point in the world,
re-picked only when the rover arrives (within 0.8 m) or when the path
to it is blocked. At a wall or corner the safety override turns the
rover toward open space (see Safety), then it drives straight again
in the new heading. It never backs up, and it has no memory of where
it has been, so it can revisit rooms.

## Map

A 20×20 m occupancy grid (5 cm cells) built from the LD19's scans.
**Scan matching** (coarse-to-fine grid search) corrects the rover's
pose against the map as it drives, so the pose doesn't drift with
speed — no wheel encoders or gyro needed. Heading is a compass
bearing, clockwise-positive (0 = north). Tap the map to set a goal;
Clear map resets the grid and pose.

## Safety (both LiDARs)

Under Tour and Go-to-goal, the **LD19 forward arc (±30°) and the
TF-Luna low front** both watch for obstacles. If either reads under
the minimum distance (slider on the page, 0.30–2.00 m, default
0.30 m), the rover stops and turns via **minimum-angle escape**: it
scans the full 360° for the nearest clear direction and turns that
way, left or right, whichever is closer. It never backs up. If both
LiDARs go offline, it stops.

**Manual driving** has its own protection: it slows down linearly
below 1.0 m from an obstacle and stops at 0.25 m. Forward uses the
LD19 forward arc + TF-Luna; reverse uses the LD19 rear arc (±45°).
Turns are unscaled.

## Reinforcement learning

`rl_rover/` is the training package (Gymnasium + Stable-Baselines3
PPO). Training runs on a Mac, not the Pi: the agent learns in a 2D
simulated room from the reward function, then the policy is exported
to ONNX and copied to the Pi (`rover_policy.onnx`, loaded lazily when
Tour or Go-to-goal is first used; needs `onnxruntime` installed for
the Pi's `/usr/bin/python3`).

- **v1** — go to a random target without crashing: +100 goal, −100
  collision, small step/progress shaping. 500k steps, **87% success**
  over 100 eval episodes.
- **v2** — adds a margin rule (no penalty while the tighter side has
  ≥ 0.25 m clearance, so it stays centered in narrow paths but not
  glued to the center in wide ones) and a small turn/stop penalty so
  it prefers driving straight. Same 27-number observation, so the
  Pi-side code is unchanged.

`HOW_TRAINING_WORKS.md` in `rl_rover/` explains the whole process in
plain English, including how to change the rewards and retrain.

## Motor map (measured)

- M1 = rear-right, M2 = rear-left, M3 = front-right, M4 = front-left.
- Negative throttle = forward.

If a wheel spins backwards, power off and swap that motor's two wires
in its HAT terminal block.

## Files

- `rover_server.py` — the drive server: page, camera, motors, LiDARs,
  map, policy, tour, safety. Run it on the Pi with
  `sudo systemctl start rover-server` (or `python3 rover_server.py`).
- `BUILD_GUIDE.md` — the full step-by-step build guide, including the
  Pi setup, the LD19 mount, and every mode.
- `PARTS_LIST.md` — chassis, motor HAT, batteries, LiDARs, and what
  you already own.
- `rover-server.service` — systemd unit so the server starts on boot.
- `ld19_gantry.py` — generates the 3D-printable LD19 mount
  (legs + feet + platform, one 180×180 mm plate). v3: flange-bolted,
  no closed slots, solid flanges (self-tapping M3 screws).
- `ld19_test.py` — standalone LD19 reader test.
- `tfluna_test.py` — standalone TF-Luna reader test.
- `tfluna_holder.py` — generates the 3D-printable TF-Luna holder.
- `rl_rover/` — the RL training package: `rover_env.py` (sim env),
  `train.py`, `test_env.py`, `README.md`, `requirements.txt`,
  `HOW_TRAINING_WORKS.md`.

## Tests

Run on a computer (no Pi hardware needed — motors and sensors are
mocked):

- `test_tour.py` — tour goal placement, fixed-point repick, blocked
  goal, hysteresis, safety override in tour.
- `test_policy.py` — policy observation matches training exactly,
  ONNX drives forward, override turns away from walls, stops at goal.
- `test_escape.py` — minimum-angle escape direction in corners.
- `test_manual_safety.py` — manual slow-down/stop distances.
- `test_port_detect.py` — LiDAR USB port auto-detection, including
  swapped ports.
- `test_heading.py` — compass-bearing heading and pose tracking.
- `test_motors.py` / `test_motor_map.py` — spin one motor at a time
  (wheels in the air) to check wiring before driving.
- `rl_rover/test_env.py` — the training environment's rewards and
  dynamics against reference math.
