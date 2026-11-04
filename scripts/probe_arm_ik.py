"""Arm IK probe for push v4 (Isaac-Push-Arms-K1-*): does the stub tip reach the IK target?

Holds a constant action (zero base command, given (u, v) per arm) and prints, per arm,
the IK target and the stub tip in the base frame, their distance, and the arm joint
positions vs limits, plus the arm actuator gains.
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Push-Arms-K1-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--uv", type=float, nargs=2, default=(0.0, 0.0))
parser.add_argument("--sweep", action="store_true")
parser.add_argument("--box_touch", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import k1_velocity.tasks.push  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from k1_velocity.tasks.push import push_arms_mdp as amdp
from k1_velocity.tasks.push import push_mdp as pmdp


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    terms = [env.action_manager.get_term(n) for n in ("arm_left", "arm_right")]
    for name, t in zip(("arm_left", "arm_right"), terms):
        jn = [robot.joint_names[i] for i in t._joint_ids]
        print(f"[{name}] joints={jn} body={t._body_name} offset={t.cfg.body_offset.pos}")
    for an, act in robot.actuators.items():
        print(f"[actuator {an}] joints={act.joint_names} kp={act.stiffness[0].tolist()} kd={act.damping[0].tolist()}")
    lim = robot.data.soft_joint_pos_limits
    lim = lim.torch if hasattr(lim, "torch") else lim
    jl = terms[0]._joint_ids
    print(f"[limits L soft] lo={lim[0, jl, 0].tolist()} hi={lim[0, jl, 1].tolist()}")
    if args_cli.box_touch:
        # box face pressed onto the trunk: body_box_contact must fire, arm links excluded
        from isaaclab.managers import SceneEntityCfg
        cfg = SceneEntityCfg("box_contact", body_names=["^(?!.*(Arm|hand)).*$"])
        cfg.resolve(env.scene)
        sensor = env.scene.sensors["box_contact"]
        print(f"[box_touch] penalized bodies={[sensor.body_names[i] for i in cfg.body_ids]}")
        box = env.scene["box"]
        rq = robot.data.root_quat_w
        rq = rq.torch if hasattr(rq, "torch") else rq
        rp = robot.data.root_pos_w
        rp = rp.torch if hasattr(rp, "torch") else rp
        half = pmdp._state(env).half_extents[:, 0]
        fwd = torch.zeros_like(rp)
        fwd[:, 0] = half + 0.08            # face 8 cm in front of the trunk origin, then pushed in
        import isaaclab.utils.math as mu
        pose = torch.cat([rp + mu.quat_apply(rq, fwd), rq], -1)
        zero_act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        for step in range(30):
            pose[:, :3] -= 0.01 * mu.quat_apply(rq, torch.tensor([[1.0, 0, 0]], device=env.device).expand_as(rp))
            box.write_root_pose_to_sim(pose)
            box.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
            env.step(zero_act)
            n = amdp.body_box_contact_count(env, cfg)
            f = sensor.data.force_matrix_w
            f = (f.torch if hasattr(f, "torch") else f)[0].norm(dim=-1).amax(-1)
            hit = [(sensor.body_names[i], round(float(f[i]), 1)) for i in range(len(f)) if f[i] > 1.0]
            print(f"box_touch step {step:2d} count={n.tolist()} env0 contacts={hit}")
        env.close()
        return
    if args_cli.sweep:
        # one left-arm config per env: where is the stub tip relative to the shoulder?
        import itertools
        grid = list(itertools.product((-1.5, -0.75, 0.0), (-1.3, 0.0), (0.0,), (-2.0, -1.0, 0.0, 1.0, 2.0)))
        n = min(len(grid), env.num_envs)
        q = robot.data.default_joint_pos
        q = (q.torch if hasattr(q, "torch") else q).clone()
        ids = torch.tensor(jl, device=env.device)
        q[:n, ids] = torch.tensor(grid[:n], device=env.device)
        zero = torch.zeros_like(q)
        robot.write_joint_state_to_sim(q, zero)
        robot.set_joint_position_target(q)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.sim.get_physics_dt())
        tips = pmdp._to_base(env, amdp.stub_tips_w(env))[:, 0]
        names = [b.split("/")[-1] for b in robot.body_names]
        bw = robot.data.body_pos_w
        bw = bw.torch if hasattr(bw, "torch") else bw
        sh = pmdp._to_base(env, bw[:, [names.index("Left_Arm_1")]])[:, 0]
        hl = pmdp._to_base(env, bw[:, [names.index("left_hand_link")]])[:, 0]
        for i in range(n):
            print(f"sweep q={grid[i]} shoulder={[round(x, 3) for x in sh[i].tolist()]} "
                  f"elbow(hand_link)={[round(x, 3) for x in (hl[i] - sh[i]).tolist()]} "
                  f"tip-shoulder={[round(x, 3) for x in (tips[i] - sh[i]).tolist()]}")
        env.close()
        return
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    act[:, 4:6] = torch.tensor(args_cli.uv)
    act[:, 6:8] = torch.tensor(args_cli.uv)
    for step in range(args_cli.steps):
        env.step(act)
        if step % 10 and step != args_cli.steps - 1:
            continue
        tips = pmdp._to_base(env, amdp.stub_tips_w(env))[0]
        for j, t in enumerate(terms):
            tgt = t.target_b[0]
            q = robot.data.joint_pos
            q = (q.torch if hasattr(q, "torch") else q)[0, t._joint_ids]
            lo, hi = lim[0, t._joint_ids, 0], lim[0, t._joint_ids, 1]
            at_lim = ((q - lo).abs() < 0.03) | ((hi - q).abs() < 0.03)
            print(f"step {step:3d} {'LR'[j]} tgt={[round(x, 3) for x in tgt.tolist()]} "
                  f"tip={[round(x, 3) for x in tips[j].tolist()]} err={float((tgt - tips[j]).norm()):.3f} "
                  f"q={[round(x, 2) for x in q.tolist()]} at_limit={at_lim.tolist()}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
