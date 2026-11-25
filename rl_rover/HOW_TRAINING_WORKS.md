# How the rover's reinforcement learning works

A plain-English explanation of the training process, with worked
examples and a walkthrough of the code. No RL background assumed.

---

## 1. The idea in one paragraph

We want the rover to drive itself to a goal point in a room without
hitting anything, staying a polite distance from walls, and taking a
reasonably direct route. Instead of writing explicit rules for every
situation ("if a wall is on the left, turn right..."), we let the
rover **learn by trial and error in a simulator**. It tries random
moves, gets a score for each move, and gradually figures out which
moves lead to good scores. After enough practice (500,000 simulated
steps), we save what it learned as a small file (an ONNX model) and
copy that file to the Pi. The Pi never trains — it just runs the
learned model, 10 times a second, to decide what to do next.

This is the same pattern as Pollen Robotics' Microduck (a biped robot
that learns to walk in simulation and runs the trained model on a
Raspberry Pi), adapted for a wheeled rover.

---

## 2. The learning loop

Every training step is the same four-part loop:

```
1. OBSERVE   The rover looks at its sensors (LiDAR scan + goal direction).
2. ACT       The policy (a small neural network) picks one action:
             forward, turn left, turn right, or stop.
3. SCORE     The simulator moves the rover, then gives a reward
             (positive for good, negative for bad).
4. LEARN     PPO (the training algorithm) nudges the policy toward
             actions that scored well and away from ones that didn't.
```

Repeat that loop 500,000 times across 8 parallel simulators. The
policy starts out random (it crashes constantly) and ends up driving
to the goal in ~87% of random rooms.

An **episode** is one attempt: the rover starts at a random spot with
a random goal, and the episode ends when it reaches the goal (+100),
crashes into something (-100), or runs out of time (300 steps = 30 s).

---

## 3. The simulated world — `rover_env.py`

The file `rover_env.py` is the simulator. It is a 2D room, 4 m wide
(x in [-2, 2]) by 3 m tall (y in [-1.5, 1.5]), with walls around the
edge and 0–3 random box obstacles inside. It is a Gymnasium
environment, which just means it has a standard `reset()` / `step()`
interface that training libraries understand.

### The rover's physics (matching the real rover)

| Quantity | Value | Where it comes from |
|---|---|---|
| Forward speed | 0.3 m/s | `FORWARD_SPEED_M_S` in `rover_server.py` |
| Turn rate | 120 deg/s | Measured: 180 deg in 1.5 s (`SPIN_180_TIME`) |
| Body radius | 0.20 m | Any LiDAR ray closer than this = a crash |
| Goal radius | 0.25 m | Within this distance = goal reached |
| Step length | 0.1 s | The policy runs at 10 Hz, same as the Pi |

Each step, a "forward" action moves the rover 0.3 × 0.1 = **0.03 m**
(3 cm) in the direction it faces. A "turn left" action rotates it
120 × 0.1 = **12 degrees** to the left. Heading is a compass bearing:
0 = north (+y), 90 = east (+x), and turning left *decreases* it —
the same convention as the Pi's `pose_loop`.

### The LiDAR scan — `_scan()`

The rover's "eyes" are a simulated LD19. `_scan()` casts **72 rays**,
5 degrees apart, ray 0 pointing straight ahead (in the heading
direction), and returns the distance to the nearest wall or box
obstacle along each ray (up to 12 m, the LD19's max range). The math
is vectorized with NumPy: for each wall, it computes where every ray
would hit that wall (`t = (wall - pos) / direction`), keeps only the
hits in front of the rover and inside the room, and takes the minimum.
Boxes use the same idea with a slab test (`_ray_aabb()`).

The 72 rays are then downsampled to **24 sectors** (`_sectors()`):
each sector is the *minimum* distance over its 3 rays (15 degrees).
Using the minimum (not the average) matters — a narrow obstacle
hiding in one ray of a sector must still be seen.

### What the rover actually sees — `_obs()`

The policy does **not** see the clean scan. It sees a noisy version
(`_noisy_scan()`): each ray gets ±0.02 m of Gaussian noise, and 1%
of rays randomly drop out (return max range). This is on purpose —
the real LD19 is noisy, so the policy must learn to be robust to it
instead of trusting any single ray.

The observation is **27 numbers**:

```
[ 24 LiDAR sectors (each / 12 m, so 0.0–1.0),
  sin(goal bearing), cos(goal bearing),   # which way the goal is, relative to heading
  goal distance / 5 m ]                   # how far away it is (capped at 1.0)
```

The bearing is *relative* to the heading: 0 means "the goal is
straight ahead", +90° means "the goal is to my right". Using sin/cos
instead of a raw angle avoids the -180/+180 wrap-around problem.
The goal distance is divided by 5 m so it stays in a similar range
to the other inputs (neural networks train better when all inputs
are roughly the same scale).

### Example observation

The rover is at (0, 0) facing north, the goal is 1.5 m to the
north-east, and there is a wall 0.5 m to its left:

```
sectors: [1.0, 1.0, ..., 0.04, ..., 1.0]   # sector 18 (left) reads 0.5/12 = 0.04
sin/cos: [0.71, 0.71]                       # goal bearing is +45 deg (north-east)
goal distance: 0.30                         # 1.5 / 5
```

The policy's job is to look at these 27 numbers and pick an action.
It has no memory — every decision is made from the current
observation alone.

---

## 4. How the rover is scored — the reward function

The reward is the *only* thing that tells the rover what "good"
means. Everything it learns comes from this. It is computed in
`step()` after each action. There are five parts:

### (a) Goal: +100

Reaching within 0.25 m of the goal ends the episode with +100.
This is the big prize — everything else is small by comparison, so
the rover always prefers reaching the goal over anything else.

### (b) Collision: -100

Any LiDAR ray closer than 0.20 m (the body radius) ends the episode
with -100. This is the big punishment — the rover learns very quickly
that crashing is the worst thing it can do.

### (c) Progress: +10 × (distance closed)

Every step, the rover gets +10 times how much closer it got to the
goal. Driving straight toward the goal at 0.3 m/s closes 0.03 m per
step, so that is **+0.3 per step**. Driving away from the goal gives
-0.3. Turning in place gives 0 (no progress). This is the dense,
step-by-step signal that guides the rover — without it, the rover
would only learn from the rare +100/-100 at the end of an episode,
which is far too sparse to learn from.

### (d) Step penalty: -0.1 per step

Every step costs -0.1, win or lose. This is a small "hurry up" tax:
a 100-step episode costs -10, so the rover prefers shorter routes.
It also stops the rover from dithering forever — standing still
accumulates -0.1 per step with no progress reward to offset it.

### (e) Side margin: -1.0 × margin violation

The rover reads its left clearance (sectors 16–20, around 270°) and
its right clearance (sectors 4–8, around 90°) and takes the smaller
of the two. If that nearer side has at least **0.25 m** of clearance,
the penalty is **0** — no need to be exactly centered; as long as it
leaves enough margin, that is good enough. If the nearer clearance
drops below 0.25 m, the penalty ramps up linearly to -1.0 at zero
clearance.

```
clearance = min(left, right)
if clearance >= 0.25:  penalty = 0
else:                  penalty = -1.0 * (0.25 - clearance) / 0.25
```

In a narrow path (say a 0.5 m corridor — a standard interior door),
being centered gives 0.25 m on each side — exactly at the margin, so
no penalty. Drifting 0.15 m to one side leaves 0.10 m on that side,
which is a penalty of -1.0 × (0.25 - 0.10) / 0.25 = **-0.60** per
step. So the rover is pushed back to center *in a narrow path*, but
in a wide path it can sit off-center freely. This is
`_margin_violation()` in the code.

### (f) Turn / stop penalties: -0.05 each

Turning left/right or stopping costs an extra -0.05 per step. A
forward step toward the goal nets +0.3 (progress) - 0.1 (step) =
**+0.2**. A turn step nets -0.1 (step) - 0.05 (turn) = **-0.15**,
with no progress. So the policy prefers to drive straight and only
turns when it has to — less stop-and-turn, shortest time wins. The
goal reward (+100) is so much larger that the policy will still turn
when turning is the way to reach the goal; it just will not turn
when driving straight would do.

### Worked example: one step's reward

The rover is centered in a 1 m-wide corridor, goal 1.5 m ahead.
It drives forward one step (0.03 m toward the goal):

```
step penalty:      -0.10
progress:          +10 × 0.03 = +0.30
margin violation:  0.00   (0.5 m clearance each side, above the 0.25 m margin)
turn/stop:         0.00   (it drove forward)
TOTAL:             +0.20
```

Same situation, but the rover is hugging the left wall (0.20 m
clearance on the left, 0.80 m on the right) and drives forward:

```
step penalty:      -0.10
progress:          +0.30
margin violation:  -1.0 × (0.25 - 0.20) / 0.25 = -0.20
turn/stop:         0.00
TOTAL:             +0.00
```

Same forward motion, same progress — but the wall-hugging step
scores **+0.00** while the centered step scores **+0.20**. Over
hundreds of steps, that difference is what teaches the rover to
keep its distance from walls.

---

## 5. How the rover learns — `train.py` and PPO

`train.py` uses **PPO** (Proximal Policy Optimization) from
Stable-Baselines3. You do not need to understand PPO's internals to
use it; here is what it does at a high level:

1. **Collect experience.** 8 copies of the environment run in
   parallel. The current policy plays in each one for 2,048 steps,
   recording (observation, action, reward) at every step.
2. **Estimate advantage.** For each step, PPO asks: "was this action
   better or worse than the average action in this situation?" An
   action that led to a higher-than-expected reward gets a positive
   advantage; one that led to a crash gets a large negative advantage.
3. **Update the policy.** PPO adjusts the neural network's weights
   a little, in the direction that makes high-advantage actions more
   likely and low-advantage actions less likely. The "proximal" part
   is a safety clip: it never changes the policy too much in one
   update, so learning is stable and does not suddenly forget
   everything.
4. **Repeat.** New experience is collected with the updated policy,
   and the cycle continues for 500,000 total steps.

The policy itself is a small neural network: **27 inputs → 64 neurons
→ 64 neurons → 4 outputs** (one score per action). The action with
the highest score is the one the rover takes. "64×64" is deliberately
tiny — it has to run on a Pi at 10 Hz, and a small network is also
less likely to overfit to the simulator.

### Hyperparameters (the knobs in `train.py`)

| Knob | Value | What it does |
|---|---|---|
| `learning_rate` | 3e-4 | How big each policy update is. Too big = unstable, too small = slow. |
| `n_steps` | 2048 | Steps collected per environment before each update. |
| `batch_size` | 64 | How many experiences are used per gradient step. |
| `n_epochs` | 10 | How many passes over the collected data per update. |
| `gamma` | 0.99 | How much future rewards matter (0.99 = a reward 100 steps away still counts ~37%). |
| `gae_lambda` | 0.95 | Smooths the advantage estimates (less noisy learning). |
| `clip_range` | 0.2 | The PPO safety clip — the policy can't change by more than 20% per update. |
| `ent_coef` | 0.01 | A small bonus for keeping the policy exploratory (trying different actions). |
| `n_envs` | 8 | Parallel simulators — 8x more experience per wall-clock second. |

### Domain randomization (sim-to-real)

Every episode randomizes: the start pose, the goal position, 0–3 box
obstacles (random size and position), LiDAR noise and dropouts, and
±10% speed / turn-rate jitter. This is the key to making the policy
work on the real rover instead of just in the simulator: the policy
never sees the same room twice, so it learns *general* skills
("steer toward the goal, keep away from close readings") instead of
memorizing one room's layout. The real world is just another random
room it has never seen.

### Evaluation — `evaluate()`

After training, the policy is tested on 100 fresh random episodes
(different seeds from training, so it cannot have memorized them).
Three numbers are reported:

- **Success rate** — the fraction of episodes where it reached the
  goal without crashing. Random policy: ~2%. v1 (500k steps): **87%**.
- **Mean episode reward** — the average total score per episode.
- **Mean margin violation** — how often it pressed against a wall
  (0 = it always kept its side margin).

The training script also saves a **rollout plot** (`rollout.png`): a
top-down drawing of one episode, so you can *see* the path the
policy took — whether it drove straight, cut corners, or hugged a
wall.

---

## 6. From training to the Pi — the ONNX export

The trained policy is a PyTorch model. The Pi cannot easily run
PyTorch (it is heavy), so `train.py` exports the policy to **ONNX**
(Open Neural Network Exchange), a portable format that ONNX Runtime
can run efficiently on the Pi.

The export wraps the policy in `OnnxablePolicy`, which returns the
**deterministic** action (argmax — the highest-scoring action, no
randomness). This matters: the Pi must behave exactly like the policy
that was evaluated, so there is no sampling or exploration at runtime.

The script then verifies the export: it feeds the same random
observation to both the PyTorch model and the ONNX file and checks
they pick the same action ("MATCH"). If it ever says MISMATCH, do
not deploy that file.

The ONNX file takes **27 floats** (the observation) and returns **1
integer** (the action: 0/1/2/3). That is the entire interface between
the training world and the Pi. The Pi-side code (`rover_server.py`,
"Go to goal" mode) builds the exact same 27 numbers from the live
LD19 scan and the map pose, runs the model, and sends the resulting
drive command — 10 times a second, with the rule-based both-LiDAR
safety override underneath so the model can never drive the rover
into a wall.

### A note on the Mac (Python 3.15)

The Mac runs Python 3.15, and `onnxruntime` has no wheel for 3.15
yet, so the ONNX *check* is skipped there (the export itself does
not need it — the `.onnx` file is still written). The Pi runs
Python 3.12/3.13, where `onnxruntime` installs normally. That is
why `train.py` wraps the ONNX Runtime import in a try/except and
why `requirements.txt` lists `onnxruntime` as optional.

---

## 7. A full worked episode

Seed 7, one episode from the v1 rollout. The room is 4×3 m with two
box obstacles. The rover starts at (-1.2, -0.8) facing east; the goal
is at (1.0, 0.9).

| Step | What happens | Reward |
|---|---|---|
| 0 | Reset. Rover at (-1.2, -0.8), goal 2.5 m away to the north-east. | — |
| 1–10 | Policy drives forward. The goal is ahead and slightly left, so it mixes forward steps with small left turns. Each forward step: +0.2 net. | ~+2.0 total |
| 11 | A box obstacle appears in the forward sectors (0.6 m ahead). The policy turns left to go around it. Turn step: -0.15. | -0.15 |
| 12–20 | It drives forward along the left side of the box, keeping ~0.5 m clearance (above the 0.25 m margin, so no margin penalty). | ~+1.8 total |
| 21–30 | It turns right to head back toward the goal, then drives straight. A few turn steps at -0.15 each, then forward steps at +0.2. | ~+0.5 total |
| 31 | It enters the goal radius (0.25 m). Episode ends. | **+100** |

Total episode reward: roughly +104. A policy that crashed into the
box at step 11 instead would have scored about +2 (progress so far)
- 100 (collision) = **-98**. That 200-point gap between success and
crash is what drives all the learning.

---

## 8. Code map

| File | What it is |
|---|---|
| `rover_env.py` | The simulator. `RoverEnv` (Gymnasium env), `_scan()` (LiDAR ray-casting), `_obs()` (27-number observation), `_margin_violation()` (side-margin penalty), `step()` (physics + reward + episode end). |
| `train.py` | The trainer. Builds 8 parallel envs, runs PPO for N timesteps, evaluates on 100 fresh episodes, saves a rollout plot, exports to ONNX and verifies it. |
| `test_env.py` | 35 checks that the simulator is correct: scan math vs. a scalar reference, dynamics (forward/turn directions), reward signs (goal/collision/progress/margin/turn/stop), and a random-policy sanity check. |
| `requirements.txt` | `gymnasium`, `stable-baselines3`, `torch`, `numpy`, `matplotlib`, `onnx` (and `onnxruntime`, optional on the Mac). |
| `README.md` | Setup, training commands, and what each output file is. |
| `runs/v1/` | The v1 trained model (`ppo_rover.zip`, `rover_policy.onnx`, `eval.txt`, `rollout.png`). |
| `runs/v2/` | The v2 retrain (margin + straight-drive rewards) — created when you run training with `--out runs/v2`. |

### The reward constants (tune these in `rover_env.py`)

| Constant | Value | Effect of changing it |
|---|---|---|
| `GOAL_REWARD` | 100.0 | The prize. Bigger = the rover cares more about reaching the goal vs. anything else. |
| `COLLISION_PENALTY` | -100.0 | The punishment. Bigger magnitude = more cautious driving. |
| `STEP_PENALTY` | -0.1 | Hurry-up tax. Bigger magnitude = shorter routes, but also more risk-taking. |
| `PROGRESS_SCALE` | 10.0 | How much closing distance matters. Bigger = more direct routes to the goal. |
| `CENTER_SCALE` | 1.0 | How much the side-margin violation matters. Bigger = more clearance from walls. |
| `SIDE_MARGIN_M` | 0.25 m | The clearance the rover tries to keep on its tighter side. Bigger = more cautious, but less able to fit through narrow gaps. |
| `TURN_PENALTY` | -0.05 | Cost of turning. Bigger magnitude = straighter paths, but slower to react to obstacles. |
| `STOP_PENALTY` | -0.05 | Cost of stopping. Bigger magnitude = less dawdling. |

**If you change any reward constant, you must retrain from scratch.**
The reward is baked into the policy during training; changing it
afterwards does not change what the already-trained policy does.

---

## 9. Common questions

**Why simulation instead of training on the real rover?**
Training needs hundreds of thousands of trial-and-error episodes, and
most early episodes end in a crash. In simulation a crash costs
nothing and an episode takes milliseconds. On the real rover a crash
could break the LiDAR, and each episode takes 30 real seconds. The
sim-to-real gap (different friction, noise, timing) is handled by
domain randomization and by the safety override on the Pi.

**Why does the policy need the goal bearing and distance?**
Without them it would just wander. The 24 LiDAR sectors tell it
where the obstacles are; the goal bearing/distance tell it where to
go. Together they are enough to decide "drive forward" vs. "turn
toward the goal" vs. "turn away from that wall".

**Why is the observation 27 numbers and not the raw 72 rays?**
The 24-sector downsampling (min per 15°) keeps the input small
enough for a tiny network to learn from, while preserving the
information that matters (the closest obstacle in each direction).
The raw 72 rays would work too, but the network would be bigger
and slower on the Pi for no real benefit.

**What if the policy drives the rover into a wall on the real Pi?**
It should not be able to: the Pi-side safety override checks both
LiDARs every step, and if either sees an obstacle under the minimum
distance it takes over and turns the rover away (minimum-angle
escape). The policy is a *suggestion*; the safety rule is a *veto*.

**How do I know if a retrain worked?**
Compare `eval.txt` before and after: success rate should stay high
(v1 was 87%), and the mean margin violation should stay low. Then
look at `rollout.png` — the path should look direct and keep its
distance from obstacles. The final test is always the real room.
