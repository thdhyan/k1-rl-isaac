"""Sim2sim: run an mjlab-trained K1 velocity checkpoint in Isaac Lab (Newton or PhysX) and record it.

The checkpoint's actor MLP + obs normaliser are loaded directly (no rsl-rl runner), so any mjlab rsl-rl
``model_*.pt`` works. Task ``Isaac-Velocity-Flat-K1-Mjlab-v0`` builds the identical 225-dim obs.

  K1_PHYSICS=newton /isaac-sim/python.sh play_mjlab_policy.py --checkpoint model_4999.pt \
      --cmd 0.4 0 0 --steps 750 --video_out out.mp4 --headless
Prints ``SIM2SIM vx_mean=.. vy_mean=.. wz_mean=.. falls=.. trunk_z=..`` (envs 0..N-1, after 1 s settle).
"""
import argparse
import inspect

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-K1-Mjlab-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=750)
parser.add_argument("--cmd", type=float, nargs=3, default=(0.4, 0.0, 0.0))
parser.add_argument("--video_out", default="")
parser.add_argument("--label", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video_out:
  args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import k1_velocity.tasks.velocity  # noqa: F401,E402


def load_mlp(ckpt, device):
  sd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
  layers, i = [], 0
  while f"mlp.{i}.weight" in sd:
    w = sd[f"mlp.{i}.weight"]
    lin = nn.Linear(w.shape[1], w.shape[0])
    lin.weight.data, lin.bias.data = w, sd[f"mlp.{i}.bias"]
    layers += [lin] + ([nn.ELU()] if f"mlp.{i + 2}.weight" in sd else [])
    i += 2
  mlp = nn.Sequential(*layers).to(device).eval()
  return mlp, sd["obs_normalizer._mean"].to(device), (sd["obs_normalizer._std"] + 1e-2).to(device)


def main():
  cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
  vx, vy, wz = args.cmd
  r = cfg.commands.twist.ranges
  r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (vx, vx), (vy, vy), (wz, wz)
  cfg.episode_length_s = args.steps * cfg.sim.dt * cfg.decimation + 1.0
  if args.video_out:
    from isaaclab_visualizers.kit import KitVisualizerCfg
    kw = dict(headless=True, eye=cfg.viewer.eye, lookat=cfg.viewer.lookat, window_width=1280, window_height=720)
    cfg.sim.visualizer_cfgs = [KitVisualizerCfg(**{k: v for k, v in kw.items()
                                                   if k in inspect.signature(KitVisualizerCfg).parameters})]
  env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video_out else None)
  base = env.unwrapped
  dev = base.device
  mlp, mean, std = load_mlp(args.checkpoint, dev)
  obs, _ = env.reset()
  print(f"OBS policy dim={obs['policy'].shape[-1]} (mjlab 225) action_dim={base.action_manager.total_action_dim}")
  robot = base.scene["robot"]
  falls = torch.zeros(base.num_envs, device=dev)
  vel, z = [], []
  frames = []
  for t in range(args.steps):
    with torch.no_grad():
      a = mlp((obs["policy"] - mean) / std)
    obs, _, term, trunc, _ = env.step(a)
    falls += term.float()
    if t >= 50:
      vel.append(torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:]], -1))
      z.append(robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2])
    if args.video_out and t % 2 == 0:
      frames.append(env.render())
  v = torch.stack(vel).mean((0, 1)).tolist()
  print(f"SIM2SIM cmd={tuple(args.cmd)} vx_mean={v[0]:.3f} vy_mean={v[1]:.3f} wz_mean={v[2]:.3f} "
        f"falls={int(falls.sum())} envs={base.num_envs} trunk_z={torch.stack(z).mean():.3f}")
  if args.video_out:
    imageio.mimsave(args.video_out, frames, fps=25)
    print(f"VIDEO {args.video_out} frames={len(frames)}")
  env.close()


main()
app.close()
