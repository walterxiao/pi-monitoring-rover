#!/usr/bin/env python3
"""Train the rover's go-to-goal policy with PPO (Stable-Baselines3) and
export it to ONNX for the Pi.

Run on the Mac (training machine):
    python3 train.py --timesteps 500000 --out runs/v1
    # ~10-20 min on Apple Silicon (MPS) or a fast CPU for a first policy.

What it does:
  1. Trains PPO on RoverEnv (8 parallel envs, small 64x64 MLP policy).
  2. Evaluates the trained policy for 100 episodes (success rate).
  3. Exports the policy to ONNX (rover_policy.onnx) — the file the Pi
     loads with ONNX Runtime. The export uses the deterministic action
     (argmax), so the Pi behaves exactly like the evaluated policy.
  4. Saves a rollout plot (rollout.png) showing the trained policy
     driving to a goal, and the reward curve (training_curve.png).

The ONNX model takes a 27-float observation and returns one int
action (0=forward, 1=left, 2=right, 3=stop) — the same commands the
drive page already sends.
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from rover_env import RoverEnv


class OnnxablePolicy(torch.nn.Module):
    """Wrap an SB3 policy so torch.onnx.export sees a single output:
    the deterministic action (argmax) for a given observation."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        return self.policy.forward(observation, deterministic=True)[0]


def evaluate(model, n_eps=100, seed=0):
    """Run the trained policy for n_eps episodes; return success rate,
    mean episode reward, and mean margin violation (0 = the rover
    kept its side margin everywhere, 1 = it pressed against a wall)."""
    env = RoverEnv()
    succ = 0
    rewards = []
    margin_vs = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        vs = []
        for _ in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            vs.append(info.get("margin_violation", 0.0))
            if term or trunc:
                break
        rewards.append(total)
        margin_vs.append(float(np.mean(vs)) if vs else 0.0)
        if info.get("success"):
            succ += 1
    return succ / n_eps, float(np.mean(rewards)), float(np.mean(margin_vs))


def save_rollout_plot(model, path, seed=7):
    """Draw one episode of the trained policy driving (top-down)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    env = RoverEnv()
    obs, _ = env.reset(seed=seed)
    xs, ys = [env.x], [env.y]
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(action))
        xs.append(env.x)
        ys.append(env.y)
        if term or trunc:
            break
    fig, ax = plt.subplots(figsize=(6, 4.5))
    hw, hh = env.room_w / 2.0, env.room_h / 2.0
    ax.set_xlim(-hw - 0.2, hw + 0.2)
    ax.set_ylim(-hh - 0.2, hh + 0.2)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((-hw, -hh), env.room_w, env.room_h,
                           fill=False, edgecolor="black", lw=2))
    for (x0, y0, x1, y1) in env.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="gray", edgecolor="black"))
    ax.add_patch(Circle((env.gx, env.gy), 0.25, facecolor="green", alpha=0.4))
    ax.plot(env.gx, env.gy, "g*", ms=14, label="goal")
    ax.plot(xs, ys, "b-", lw=1.5, label="rover path")
    ax.plot(xs[0], ys[0], "bo", ms=8, label="start")
    ax.plot(xs[-1], ys[-1], "rs", ms=8, label="end")
    ax.set_title("Trained policy rollout (seed 7)")
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--eval-eps", type=int, default=100)
    ap.add_argument("--no-onnx", action="store_true",
                    help="skip the ONNX export (training only)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device: {device} | timesteps: {args.timesteps} | envs: {args.n_envs}")

    def make_env():
        return Monitor(RoverEnv())

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
    model = PPO(
        "MlpPolicy", vec_env,
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=args.seed,
        device=device,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    print(f"training done in {time.time() - t0:.0f} s")
    model.save(os.path.join(args.out, "ppo_rover"))
    print(f"saved model to {args.out}/ppo_rover.zip")

    succ, mean_r, mean_margin = evaluate(model, n_eps=args.eval_eps)
    print(f"EVAL: success rate {succ:.0%} over {args.eval_eps} episodes "
          f"| mean episode reward {mean_r:.1f} (random policy: ~2%) "
          f"| mean margin violation {mean_margin:.2f} "
          f"(0 = kept its side margin, 1 = pressed against a wall)")
    with open(os.path.join(args.out, "eval.txt"), "w") as f:
        f.write(f"timesteps={args.timesteps}\n")
        f.write(f"success_rate={succ:.3f}\n")
        f.write(f"mean_reward={mean_r:.1f}\n")
        f.write(f"mean_margin_violation={mean_margin:.3f}\n")

    save_rollout_plot(model, os.path.join(args.out, "rollout.png"))
    print(f"saved rollout plot to {args.out}/rollout.png")

    if not args.no_onnx:
        onnx_path = os.path.join(args.out, "rover_policy.onnx")
        # ONNX export runs on CPU (the exporter does not accept an MPS
        # model fed a CPU tensor): export from a CPU copy of the policy
        # so the trained model itself is untouched.
        cpu_policy = copy.deepcopy(model.policy).to("cpu")
        dummy = torch.zeros(1, 27)
        torch.onnx.export(
            OnnxablePolicy(cpu_policy),
            dummy, onnx_path,
            input_names=["observation"],
            output_names=["action"],
            opset_version=14,
            dynamo=False,
        )
        print(f"exported ONNX policy to {onnx_path}")
        # Sanity-check the ONNX file with ONNX Runtime, if installed.
        # (onnxruntime has no wheel for Python 3.15 yet, so it is
        # optional on the Mac — the export itself does not need it.
        # The Pi, which runs the model, uses its own Python 3.12/3.13
        # where onnxruntime installs fine.)
        try:
            import onnxruntime as ort
        except ImportError:
            print("ONNX check skipped: onnxruntime not installed "
                  "(optional; the .onnx file was still exported)")
        else:
            sess = ort.InferenceSession(onnx_path)
            rng = np.random.default_rng(0)
            obs = rng.uniform(-1, 1, size=(1, 27)).astype(np.float32)
            onnx_action = sess.run(None, {"observation": obs})[0]
            torch_action, _ = model.predict(obs[0], deterministic=True)
            print(f"ONNX check: onnx={int(onnx_action[0])} "
                  f"pytorch={int(torch_action)} "
                  f"({'MATCH' if int(onnx_action[0]) == int(torch_action) else 'MISMATCH'})")


if __name__ == "__main__":
    main()
