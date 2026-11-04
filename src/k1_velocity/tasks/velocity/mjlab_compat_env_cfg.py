"""Isaac Lab twin of mjlab's ``Mjlab-Velocity-Flat-Booster-K1``: runs mjlab-trained K1 checkpoints unchanged.

Robot = ``K1_MJLAB_CFG`` (USD exported from the mjlab model, soft Booster gains). Actor obs is exactly the
mjlab one, 225 dims: 5-step term-major history of
  base_ang_vel 3 | projected_gravity 3 | joint_pos_rel 12 | joint_vel_rel 12 | last_action 12 | command 3
with legs in model order ``LEG_JOINTS`` (L x6, R x6), which is also the action order. Action =
default + K1_MJLAB_ACTION_SCALE * a. Upper body holds its home pose (mjlab drives it randomly in training).
dt 0.005, decimation 4 (50 Hz). Physics backend from ``K1_PHYSICS`` (newton default, physx opt-in).
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from k1_velocity.assets_mjlab import K1_MJLAB_ACTION_SCALE, K1_MJLAB_CFG, LEG_JOINTS, UPPER_REGEX
from k1_velocity.sim_backend import apply_physics_backend, ensure_physx_gpu_capacity

from .velocity_env_cfg import mdp

LEGS = SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)


def hold_upper_body(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=UPPER_REGEX)):
  """Upper body has no action term: pin its PD targets to the home pose."""
  robot = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  robot.set_joint_position_target(robot.data.default_joint_pos[env_ids][:, ids], joint_ids=ids, env_ids=env_ids)


@configclass
class SceneCfg(InteractiveSceneCfg):
  terrain = TerrainImporterCfg(
    prim_path="/World/ground", terrain_type="plane", collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0))
  robot: ArticulationCfg = K1_MJLAB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
  light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=900.0))


@configclass
class ObservationsCfg:
  @configclass
  class PolicyCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEGS})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": LEGS})
    actions = ObsTerm(func=mdp.last_action)
    command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})

    def __post_init__(self):
      self.enable_corruption = False
      self.concatenate_terms = True
      self.history_length = 5
      self.flatten_history_dim = True

  policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
  joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=LEG_JOINTS, preserve_order=True, scale=K1_MJLAB_ACTION_SCALE,
    use_default_offset=True)


@configclass
class CommandsCfg:
  twist = mdp.UniformVelocityCommandCfg(
    asset_name="robot", resampling_time_range=(1e9, 1e9), rel_standing_envs=0.0, heading_command=False,
    debug_vis=False,
    ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.4, 0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)))


@configclass
class EventCfg:
  reset_base = EventTerm(
    func=mdp.reset_root_state_uniform, mode="reset",
    params={"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}})
  reset_joints = EventTerm(
    func=mdp.reset_joints_by_offset, mode="reset", params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})
  hold_upper = EventTerm(func=hold_upper_body, mode="reset")


@configclass
class RewardsCfg:
  pass


@configclass
class TerminationsCfg:
  time_out = DoneTerm(func=mdp.time_out, time_out=True)
  fell = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.22})


@configclass
class K1MjlabFlatEnvCfg(ManagerBasedRLEnvCfg):
  scene: SceneCfg = SceneCfg(num_envs=16, env_spacing=3.0)
  observations: ObservationsCfg = ObservationsCfg()
  actions: ActionsCfg = ActionsCfg()
  commands: CommandsCfg = CommandsCfg()
  events: EventCfg = EventCfg()
  rewards: RewardsCfg = RewardsCfg()
  terminations: TerminationsCfg = TerminationsCfg()
  viewer: ViewerCfg = ViewerCfg(
    eye=(2.0, -2.0, 1.0), lookat=(0.0, 0.0, 0.4), origin_type="asset_root", asset_name="robot", env_index=0)

  def __post_init__(self):
    super().__post_init__()
    self.sim.use_newton_actuators = False
    apply_physics_backend(self)
    ensure_physx_gpu_capacity(self)
    self.sim.dt = 0.005
    self.decimation = 4
    self.sim.render_interval = self.decimation
    self.episode_length_s = 20.0
