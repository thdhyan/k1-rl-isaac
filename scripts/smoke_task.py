"""Build one K1 Isaac Lab task, step it with zero actions, report shapes and trunk height.

  K1_PHYSICS=newton /isaac-sim/python.sh smoke_task.py --task Isaac-Kick-Ball-K1-v0 --headless
Prints ``SMOKE <task> OK obs=.. act=.. trunk_z=.. falls=..`` or ``SMOKE <task> FAIL <error>``.
Zero action = hold the default pose, so trunk_z ~0.5 and few falls means the robot asset stands.
"""
import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import k1_velocity.tasks  # noqa: F401,E402
import k1_velocity.tasks.carry  # noqa: F401,E402
import k1_velocity.tasks.partial  # noqa: F401,E402
import k1_velocity.tasks.skills  # noqa: F401,E402


def main():
  cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
  env = gym.make(args.task, cfg=cfg)
  base = env.unwrapped
  obs, _ = env.reset()
  shapes = {k: tuple(v.shape) for k, v in obs.items()} if isinstance(obs, dict) else tuple(obs.shape)
  act = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
  falls = 0
  for _ in range(args.steps):
    _, _, term, _, _ = env.step(act)
    falls += int(term.sum())
  robot = base.scene["robot"] if "robot" in base.scene.keys() else base.scene["robot_0"]
  z = (robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]).mean()
  print(f"SMOKE {args.task} OK obs={shapes} act={base.action_manager.total_action_dim} "
        f"trunk_z={z:.3f} falls={falls}/{base.num_envs}")
  env.close()


try:
  main()
except Exception as e:  # report and keep the batch going
  traceback.print_exc()
  print(f"SMOKE {args.task} FAIL {type(e).__name__}: {str(e)[:300]}")
app.close()
