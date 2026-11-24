# Rover RL — train a go-to-goal policy in simulation, deploy it on the Pi

The Microduck pattern, adapted for this rover: **train in simulation on
the Mac, run the trained model on the Pi.** The Pi never trains — it
just runs the exported ONNX file with ONNX Runtime.

## What this is

- `rover_env.py` — a Gymnasium environment: a 2D room (4 m x 3 m) with
  walls and box obstacles, a 360 deg LiDAR scan (downsampled to 24
  sectors, the way the real LD19 scan will be fed in), and a goal point.
  The rover drives to the goal without hitting anything.
  - State (27 numbers): 24 LiDAR sectors + goal bearing (sin, cos) +
    goal distance.
  - Actions: 0 = forward, 1 = turn left, 2 = turn right, 3 = stop —
    the same commands the drive page already sends.
  - Dynamics match the real rover: 0.3 m/s forward, 120 deg/s turn,
    heading is a compass bearing (0 = north, left turn decreases it).
  - Domain randomization every episode (sim-to-real): random start,
    random goal, 0-3 random box obstacles, LiDAR noise + dropouts, and
    +/-10% speed / turn-rate jitter, so the policy is robust instead
    of memorizing one room.
  - Rewards: +100 goal, -100 collision, -0.1/step, +10 x progress
    toward the goal, and two shaping terms (v2):
    - **Side margin:** the rover reads its left/right side
      clearance from the LiDAR and is penalized only when the
      nearer side gets closer than 0.35 m. It does NOT need to be
      exactly centered — as long as it leaves enough margin, that
      is good enough. In a narrow path, any drift off-center
      violates the margin, so it stays centered there.
      Tune with `SIDE_MARGIN_M` and `CENTER_SCALE` in `rover_env.py`.
    - **Shortest time wins:** turning (-0.05) and stopping (-0.05)
      cost a little extra per step, so the policy prefers to drive
      straight and only turns when it needs to — less stop-and-turn.
      Tune with `TURN_PENALTY` and `STOP_PENALTY` in `rover_env.py`.
- `train.py` — trains PPO (Stable-Baselines3), evaluates it, exports
  it to ONNX, and saves a rollout plot.
- `test_env.py` — 35 checks that the simulator is correct (scan math
  vs. a scalar reference, dynamics, rewards, collision, goal,
  margin violation, and turn/stop penalties).
- `HOW_TRAINING_WORKS.md` — a plain-English explanation of the whole
  training process, with worked examples and a code walkthrough.

## Setup on the Mac

```bash
cd rl_rover
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This needs Python 3.11+. The `torch` download is large (~200-800 MB).
On Apple Silicon, PyTorch uses MPS (Metal) automatically — no CUDA
needed, which is why this stack works on a Mac where Microduck's
mjlab training (which requires a CUDA GPU) does not.

## Train

```bash
python3 train.py --timesteps 500000 --out runs/v2
```

- 500,000 steps is a good first run. Expect roughly 10-20 minutes on
  Apple Silicon. (For reference: 50,000 steps took 160 s on a slow
  1.5-CPU cloud VM, and reached 77% success.)
- Use `--timesteps 50000` for a quick smoke test (~2-3 min).
- Use `--n-envs 8` (default) for 8 parallel simulators.
- **v2 retrain (margin + straight drive):** the v1 policy (runs/v1)
  was trained without the margin term or the turn/stop penalties.
  To get the new behavior, retrain from scratch with the same
  command above (use `--out runs/v2`). The reward changed, so the
  old policy can't be fine-tuned — it needs a fresh 500k-step run.
  Expect a similar success rate (~85%+) with a low mean margin
  violation (reported in eval.txt and in the training output).

## What you get (in `runs/v2/`)

- `ppo_rover.zip` — the trained PyTorch model (for retraining / inspection).
- `rover_policy.onnx` — the deployable policy. Takes 27 floats, returns
  one int action. This is the file the Pi will load.
- `eval.txt` — success rate over 100 random episodes + mean reward +
  mean margin violation (0 = the rover kept its side margin, 1 =
  pressed against a wall). (Random policy: ~2% success. The 50k-step
  smoke run: 77%.)
- `rollout.png` — a top-down plot of the trained policy driving to a
  goal, so you can see it steering around obstacles.

The training script also prints an ONNX sanity check: the ONNX model
must give the same action as the PyTorch model on a fixed input
("MATCH"). If it ever says MISMATCH, do not deploy that file.

Note: you may see a DeprecationWarning about the "legacy
TorchScript-based ONNX export". It is harmless — the export still
works and the ONNX check verifies the file.

## Verify the simulator (optional)

```bash
python3 test_env.py
```

All 32 checks should PASS. This confirms the ray-casting, the
dynamics, and the rewards (including centering) against independent
reference math.

## Next step (not built yet): run it on the Pi

The Pi side is not in this package yet. When you want it, the work is:

1. Copy `rover_policy.onnx` to the Pi and `pip install onnxruntime`
   there (or `sudo apt install python3-onnxruntime`).
2. Add a policy thread to `rover_server.py`: read the live LD19 scan,
   build the same 27-number observation, run the ONNX model, and send
   the drive command — with the rule-based obstacle avoid kept as a
   safety net underneath, so the model can never drive the rover into
   a wall.
3. Test it in a real room against the rule-based avoid.

## Honest limits

- The policy only knows the simulated room. The sim-to-real gap is
  real: different friction, sensor noise, and timing on the physical
  rover. Domain randomization helps, but the first real-room test may
  still be clumsy — that is normal, and the fix is more randomization
  and more training, not a different approach.
- This is a first task (go to a goal without hitting anything). Line
  following, "go to a named room", and multi-goal routes are later
  tasks that reuse the same setup.
