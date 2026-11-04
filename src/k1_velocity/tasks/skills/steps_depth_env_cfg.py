"""Climb steps residual, DEPTH-CAMERA actor (``Isaac-Steps-Depth-Residual-K1-v0``).

Same world, action, rewards, terminations and curriculum as ``steps_env_cfg`` (height-scan
variant). Only the actor's input changes:

- actor  (``depth_actor``): deployable proprio (no base_lin_vel, no height scan) + a 24x32
  depth image from a head camera pitched 45 deg down, clipped to 2 m and flattened (768).
- critic (``teacher``): the privileged 237-dim group with the 187-point height scan
  (asymmetric actor-critic), so value learning does not wait on the image.

The camera sits on ``Head_2`` (head pitch link); the residual does not drive the head, so the
head stays at its default pose and the camera view is fixed relative to the trunk, as it would
be on the robot with the head locked. Needs ``--enable_cameras``; use ~1024 envs.
"""
import isaaclab.sim as sim_utils
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from k1_velocity.assets_mjlab import K1_HEAD_LINK
from k1_velocity.tasks.velocity import gait_clock
from k1_velocity.tasks.velocity.velocity_env_cfg import K1_LEG_JOINTS, K1RoughSceneCfg, ObservationsCfg, mdp

from .steps_env_cfg import K1StepsResidualEnvCfg

DEPTH_H, DEPTH_W = 24, 32
DEPTH_MAX = 2.0  # m; stairs in front of the feet are within ~1.5 m


def depth_image(env, sensor_cfg: SceneEntityCfg, max_depth: float = DEPTH_MAX) -> torch.Tensor:
    """Flattened depth (n, H*W) in [0, 1]: distance to image plane / max_depth, misses -> 1."""
    d = env.scene.sensors[sensor_cfg.name].data.output["distance_to_image_plane"]
    d = d.torch if hasattr(d, "torch") else d
    d = torch.nan_to_num(d, nan=max_depth, posinf=max_depth, neginf=0.0)
    return (d.clamp(0.0, max_depth) / max_depth).reshape(d.shape[0], -1)


@configclass
class StepsDepthSceneCfg(K1RoughSceneCfg):
    depth_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + K1_HEAD_LINK,
        offset=CameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.05),
            rot=(0.0, 0.38268, 0.0, 0.92388),  # xyzw: +45 deg about y = look 45 deg down
            convention="world",
        ),
        height=DEPTH_H,
        width=DEPTH_W,
        update_period=0.0,  # every env step (render runs once per decimation)
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.0, horizontal_aperture=2.0,  # ~90 deg horizontal FOV
            clipping_range=(0.05, 10.0),
        ),
    )


@configclass
class StepsDepthObservationsCfg(ObservationsCfg):
    @configclass
    class DepthActorCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)})
        actions = ObsTerm(func=mdp.last_action)
        phase_clock = ObsTerm(func=gait_clock.phase_clock, noise=None)
        depth = ObsTerm(func=depth_image, params={"sensor_cfg": SceneEntityCfg("depth_cam")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    depth_actor: DepthActorCfg = DepthActorCfg()


@configclass
class K1StepsDepthResidualEnvCfg(K1StepsResidualEnvCfg):
    scene: StepsDepthSceneCfg = StepsDepthSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: StepsDepthObservationsCfg = StepsDepthObservationsCfg()


@configclass
class K1StepsDepthResidualPlayEnvCfg(K1StepsDepthResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        gen = self.scene.terrain.terrain_generator
        gen.num_rows, gen.num_cols, gen.curriculum = 5, 4, False
        self.scene.terrain.max_init_terrain_level = None
        self.events.push_robot = None
