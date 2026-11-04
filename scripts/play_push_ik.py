"""Closed-loop Mjlab-PushIK checkpoint in Isaac Lab (K1_PHYSICS=physx|newton): the deploy controller drives two K1s
and one box, physics decides the outcome. Same scenario ranges as tasks_push_ik.reset_scene at full curriculum. One box
size in all envs (replicate_physics=True, Newton cannot clone per-env sizes for this scene).

  K1_PHYSICS=physx /isaac-sim/python.sh play_push_ik.py --checkpoint model_6998.pt --base walk_still_v2.pt \
      --steps 3000 --video_out out.mp4 --headless
Prints ``EPISODE env=.. <outcome> t=.. err=..`` and ``PUSHIK physics=.. episodes=.. success=.. ...``. Video = the env at the world origin.
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--base", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--video_out", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video_out:
  args.enable_cameras = True
app = AppLauncher(args).app

import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from k1_velocity.assets_mjlab import K1_MJLAB_CFG  # noqa: E402
from k1_velocity.sim_backend import apply_physics_backend  # noqa: E402

from k1_runtime.push_ik_core import PushIKController, box_errors  # noqa: E402
from k1_runtime import wbc  # noqa: E402
import k1_description  # noqa: E402

HALF = (0.275, 0.55, 0.22)  # mid of the training size ranges
BOX_KG, BOX_MU = 3.5, 0.45  # mid of the training ranges
TIME_S, OOB_R, TIP, FELL, SUCCESS_V = 30.0, 5.0, math.radians(30.0), math.radians(70.0), 0.15
USD = os.environ.get("K1_SOLID_USD") or str(k1_description.robot_file(k1_description.USD_MJLAB_SOLID))


def k1(path: str):
  # training's manip robot: arms kp 40 kd 2 (assets_mjlab has the 4 Hz gains); arms start in the deploy pose
  acts = dict(K1_MJLAB_CFG.actuators)
  acts["arms"] = ImplicitActuatorCfg(joint_names_expr=[".*Shoulder.*", ".*Elbow.*"], stiffness=40.0, damping=2.0,
                                     armature=0.001, effort_limit_sim=14.0, velocity_limit_sim=100.0, friction=0.0,
                                     dynamic_friction=0.0, viscous_friction=0.0)
  jp = {n: float(v) for n, v in zip(wbc.JOINTS, wbc.HOME)}
  jp.update({n: float(v) for n, v in zip(wbc.JOINTS[2:10], wbc.ARM_POSE)})
  return K1_MJLAB_CFG.replace(prim_path=path, actuators=acts, spawn=K1_MJLAB_CFG.spawn.replace(usd_path=USD),
                              init_state=K1_MJLAB_CFG.init_state.replace(joint_pos=jp))


@configclass
class SceneCfg(InteractiveSceneCfg):
  num_envs: int = 8
  env_spacing: float = 16.0
  replicate_physics: bool = True
  # a scene field, not spawned in _setup_scene: Newton misses a ground added there and everything free-falls
  terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", physics_material=sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0, dynamic_friction=1.0))
  robot_a = k1("/World/envs/env_.*/RobotA")
  robot_b = k1("/World/envs/env_.*/RobotB")
  box = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Box",
    spawn=sim_utils.CuboidCfg(
      size=tuple(2 * h for h in HALF), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
      mass_props=sim_utils.MassPropertiesCfg(mass=BOX_KG), collision_props=sim_utils.CollisionPropertiesCfg(),
      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=BOX_MU, dynamic_friction=BOX_MU),
      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.45, 0.15))),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HALF[2] + 0.005)))


@configclass
class EnvCfg(DirectRLEnvCfg):
  decimation = 4
  episode_length_s = 1e6  # outcomes are decided in main()
  action_space = 44
  observation_space = 1
  state_space = 0
  sim: SimulationCfg = SimulationCfg(dt=0.005)
  scene: SceneCfg = SceneCfg()


class PushEnv(DirectRLEnv):
  """Thin physics stepper: action = 22 joint targets per robot, in wbc.JOINTS order."""

  def _setup_scene(self):
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

  def _pre_physics_step(self, actions):
    self.targets = actions

  def _apply_action(self):
    for k, r in enumerate((self.scene["robot_a"], self.scene["robot_b"])):
      r.set_joint_position_target_index(target=self.targets[:, 22 * k:22 * (k + 1)], joint_ids=r.wbc_ids)

  def _get_observations(self):
    return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

  def _get_rewards(self):
    return torch.zeros(self.num_envs, device=self.device)

  def _get_dones(self):
    z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return z, z

  def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)


def quat_z(y):
  """xyzw, the Isaac Lab 3 convention."""
  return torch.stack([0 * y, 0 * y, torch.sin(y / 2), torch.cos(y / 2)], -1)


def yaw_of(q):
  return torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))


def scenario(env, ids, goals, H):
  """tasks_push_ik.reset_scene at p = 1, in the world frame of each env origin."""
  n, dev = len(ids), env.device
  r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)      # noqa: E731
  sym = lambda a: a * (2.0 * torch.rand(n, device=dev) - 1.0)        # noqa: E731
  org = env.scene.env_origins[ids]
  th, xy, yaw = sym(math.pi), torch.stack([sym(0.5), sym(0.5)], -1), sym(math.pi)
  dist = r(0.4, 2.5)
  goals[ids, :2] = org[:, :2] + xy + dist[:, None] * torch.stack([th.cos(), th.sin()], -1)
  goals[ids, 2] = yaw + sym(math.pi / 2)
  box = env.scene["box"]
  p = torch.cat([org + torch.cat([xy, torch.full((n, 1), H[2] + 0.005, device=dev)], -1), quat_z(yaw)], -1)
  box.write_root_pose_to_sim_index(root_pose=p, env_ids=ids)
  box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
  rad = math.hypot(H[0], H[1]) + r(0.45, 1.45)
  ang, gap = sym(math.pi), (0.75 / rad).clamp_max(math.pi) + r(0.0, 0.5)
  for k, name in enumerate(("robot_a", "robot_b")):
    rb = env.scene[name]
    al = ang + k * gap
    pos = xy + torch.stack([rad * al.cos(), rad * al.sin()], -1)
    head = torch.atan2(xy[:, 1] - pos[:, 1], xy[:, 0] - pos[:, 0]) + sym(math.pi)
    root = torch.cat([org[:, :2] + pos, org[:, 2:] + rb.data.default_root_pose.torch[ids, 2:3] + 0.01, quat_z(head)], -1)
    rb.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
    rb.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
    rb.write_joint_position_to_sim_index(position=rb.data.default_joint_pos.torch[ids], env_ids=ids)
    rb.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(rb.data.default_joint_vel.torch[ids]), env_ids=ids)


def main():
  cfg = EnvCfg()
  cfg.scene.num_envs = args.num_envs
  backend = apply_physics_backend(cfg)
  if args.video_out:
    from isaaclab_visualizers.kit import KitVisualizerCfg
    cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, eye=(4.0, 4.0, 3.5), lookat=(0.0, 0.0, 0.3),
                                                window_width=1280, window_height=720)]
  env = PushEnv(cfg, render_mode="rgb_array" if args.video_out else None)
  dev = env.device
  a, b, box = env.scene["robot_a"], env.scene["robot_b"], env.scene["box"]
  print("JOINTS", a.joint_names, flush=True)
  a.wbc_ids = b.wbc_ids = [a.joint_names.index(j) for j in wbc.JOINTS]
  ids_w = torch.tensor(a.wbc_ids, device=dev)
  N = env.num_envs
  # the camera looks at the world origin: film and mark the env that sits there
  vid = int(env.scene.env_origins[:, :2].norm(dim=-1).argmin())
  ctrl = [PushIKController(args.checkpoint, HALF) for _ in range(N)]
  loco = wbc.LocoModule(args.base, 2 * N)
  goals = torch.zeros(N, 3, device=dev)
  marker = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/goal", markers={"goal": sim_utils.CuboidCfg(
    size=(2 * HALF[0], 2 * HALF[1], 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)))}))
  env.reset()
  all_ids = torch.arange(N, device=dev)
  scenario(env, all_ids, goals, HALF)
  env.scene.write_data_to_sim()
  home = torch.tensor(wbc.HOME)
  home[2:10] = torch.tensor(wbc.ARM_POSE)
  targets = home.repeat(2 * N, 1)
  t_ep = torch.zeros(N, dtype=torch.long)
  out, secs, frames, hands = [], [], [], 0
  gravity = torch.tensor((0.0, 0.0, -1.0), device=dev)
  from isaaclab.utils.math import quat_apply_inverse

  def pose(asset):
    p, q = asset.data.root_pos_w.torch, asset.data.root_quat_w.torch
    return torch.cat([p[:, :2] - env.scene.env_origins[:, :2], yaw_of(q)[:, None]], -1).cpu().numpy()

  for step in range(args.steps):
    pa, pb, pc = pose(a), pose(b), pose(box)
    g = goals.cpu().numpy().copy()
    g[:, :2] -= env.scene.env_origins[:, :2].cpu().numpy()
    with torch.no_grad():
      cmds, dhs, arms = zip(*(ctrl[i].step(pa[i], pb[i], pc[i], g[i]) for i in range(N)))
      cmds, dhs, arms = np.stack(cmds), np.stack(dhs), np.stack(arms)
      rob = (a, b)
      q = torch.cat([r.data.joint_pos.torch[:, ids_w] for r in rob]).cpu()
      dq = torch.cat([r.data.joint_vel.torch[:, ids_w] for r in rob]).cpu()
      w = torch.cat([r.data.root_ang_vel_b.torch for r in rob]).cpu()
      qu = torch.cat([r.data.root_quat_w.torch for r in rob])
      gr = quat_apply_inverse(qu, gravity.expand(2 * N, 3)).cpu()
      # env order [a0..aN-1, b0..bN-1]; controllers give (env, agent, ...)
      c = torch.tensor(cmds, dtype=torch.float32).permute(1, 0, 2).reshape(2 * N, 3)
      legs = loco(wbc.Proprio(w, gr, q, dq),
                  wbc.LocoCmd(c[:, 0], c[:, 1], c[:, 2], torch.tensor(dhs, dtype=torch.float32).T.reshape(2 * N)))
    tgt = targets.clone()
    tgt[:, 2:10] = torch.tensor(arms, dtype=torch.float32).permute(1, 0, 2).reshape(2 * N, 8)
    tgt[:, wbc.LEGS] = legs
    tgt = tgt.reshape(2, N, 22).permute(1, 0, 2).reshape(N, 44).to(dev)
    # one env step is 4 physics steps; DirectRLEnv would overwrite nothing: dones are always False
    env.step(tgt)
    t_ep += 1
    if os.environ.get("K1_DBG") and step % 50 == 0:
      print(f"DBG {step} a={pa[vid].round(2)} b={pb[vid].round(2)} box={pc[vid].round(2)} goal={g[vid].round(2)} "
            f"cmd={cmds[vid].round(2).tolist()} dh={dhs[vid].round(2)} arm0={arms[vid, 0].round(2)} "
            f"q_arm0={q[vid, 2:10].numpy().round(2)} az={a.data.root_pos_w.torch[vid, 2].item():.3f} "
            f"boxz={box.data.root_pos_w.torch[vid, 2].item():.3f}", flush=True)

    if args.video_out and step % 2 == 0:
      marker.visualize(translations=torch.cat([goals[vid:vid + 1, :2], torch.tensor([[0.006]], device=dev)], -1),
                       orientations=quat_z(goals[vid:vid + 1, 2]))
      frames.append(env.render())
    pc = pose(box)
    vb = box.data.root_lin_vel_w.torch[:, :2].norm(dim=-1)
    tilt = torch.acos((1 - 2 * (box.data.root_quat_w.torch[:, 0] ** 2 + box.data.root_quat_w.torch[:, 1] ** 2)).clamp(-1, 1))
    fell = [torch.acos((-quat_apply_inverse(r.data.root_quat_w.torch, gravity.expand(N, 3))[:, 2]).clamp(-1, 1)) > FELL for r in rob]
    done = []
    for i in range(N):
      d, e = box_errors(pc[i], g[i])
      why = ("fell" if fell[0][i] or fell[1][i] else "tip" if tilt[i] > TIP else "oob" if math.hypot(*pc[i][:2]) > OOB_R
             else "success" if d < 0.2 and e < math.radians(10) and vb[i] < SUCCESS_V
             else "timeout" if t_ep[i] * 0.02 >= TIME_S else None)
      if why:
        print(f"EPISODE env={i} {why} t={t_ep[i] * 0.02:.1f}s err={d:.2f}m {math.degrees(e):.1f}deg", flush=True)
        out.append(why)
        secs.append(t_ep[i].item() * 0.02)
        done.append(i)
    if done:
      ids = torch.tensor(done, device=dev)
      scenario(env, ids, goals, HALF)
      env.scene.write_data_to_sim()
      for i in done:
        ctrl[i].reset()
        t_ep[i] = 0
      loco.reset(done + [i + N for i in done])
  n = max(len(out), 1)
  won = [s for s, w in zip(secs, out) if w == "success"]
  print(f"PUSHIK physics={backend} episodes={len(out)} success={out.count('success') / n:.3f} "
        f"fell={out.count('fell') / n:.3f} tip={out.count('tip') / n:.3f} oob={out.count('oob') / n:.3f} "
        f"timeout={out.count('timeout') / n:.3f} time_to_goal={np.mean(won) if won else float('nan'):.1f}s", flush=True)
  if args.video_out:
    imageio.mimsave(args.video_out, frames, fps=25)
    print(f"VIDEO {args.video_out} frames={len(frames)}", flush=True)
  sys.stdout.flush()
  os._exit(0)  # app.close() hangs after the writer on this image


main()
