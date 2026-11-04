"""Climb steps as a RESIDUAL over the frozen velocity student (PLAN_HRL_CONTROLLER.md 6b).

``Isaac-Steps-Residual-K1-v0``. World = rough velocity env with a stairs-only terrain:
pyramid stairs (down from the centre) and inverted pyramids (up from the centre), step
height growing by terrain row 4 -> 14 cm, advanced by ``terrain_levels_vel``.

    leg target = default + clip(student(vx, vy, wz) + residual_scale * a, +-1)

The frozen walk student is blind; the residual sees the teacher obs group (237: noise-free
proprio + 187-point height scan), so it has to learn the extra foot lift and the step
placement from the scan. Distill to a blind/depth student later (plan 6b).

The rough env's world-frame root_height termination (0.35 m) is wrong on stairs (an
upper step lifts the trunk, a lower one ends the episode while walking), so it is
replaced by the terrain-relative check from lower_body_mdp.
"""
import os

import isaaclab.terrains as terrain_gen
import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from k1_velocity.tasks.velocity import lower_body_mdp
from k1_velocity.tasks.velocity.velocity_env_cfg import K1VelocityRoughEnvCfg, TerminationsCfg, mdp

from . import skills_mdp as smdp
from .residual_action import FrozenStudentResidualActionCfg

STUDENT_PATH = os.environ.get("SKILL_STUDENT_POLICY", "models/k1_velocity_student_gaitfix.pt")
RESIDUAL_SCALE = 0.3  # stair lift needs more than track/kick's 0.1, less than squat's 0.5
FEET = ["left_foot_link", "right_foot_link"]

STEPS_SUB_TERRAINS = {
    "stairs_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
        proportion=0.5, step_height_range=(0.04, 0.14), step_width=0.30,
        platform_width=2.0, border_width=0.25),
    "stairs_down": terrain_gen.HfPyramidStairsTerrainCfg(
        proportion=0.4, step_height_range=(0.04, 0.14), step_width=0.30,
        platform_width=2.0, border_width=0.25),
    # keep some flat so the residual does not forget plain walking
    "flat": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.1, noise_range=(0.0, 0.01), noise_step=0.005, border_width=0.25),
}


def feet_stumble(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 per foot whose horizontal contact force exceeds 2x its vertical one (toe hits a riser)."""
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (f[..., :2].norm(dim=-1) > 2.0 * f[..., 2].abs()).float().sum(dim=1)


@configclass
class StepsResidualActionsCfg:
    joint_pos = FrozenStudentResidualActionCfg(
        student_path=STUDENT_PATH, residual_scale=RESIDUAL_SCALE, head=False, command_name="base_velocity"
    )


@configclass
class StepsResidualRewardsCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=10.0,
                                   params={"command_name": "base_velocity", "std": 0.25})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=3.0,
                                  params={"command_name": "base_velocity", "std": 0.25})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"])})
    feet_stumble = RewTerm(func=feet_stumble, weight=-1.0,
                           params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
    residual_l2 = RewTerm(func=smdp.residual_l2, weight=-0.05, params={"action_name": "joint_pos"})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class StepsResidualTerminationsCfg(TerminationsCfg):
    root_height = DoneTerm(func=lower_body_mdp.root_height_below_terrain,
                           params={"minimum_height": 0.35, "sensor_name": "height_scanner"})


@configclass
class StepsResidualCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class K1StepsResidualEnvCfg(K1VelocityRoughEnvCfg):
    actions: StepsResidualActionsCfg = StepsResidualActionsCfg()
    rewards: StepsResidualRewardsCfg = StepsResidualRewardsCfg()
    terminations: StepsResidualTerminationsCfg = StepsResidualTerminationsCfg()
    curriculum: StepsResidualCurriculumCfg = StepsResidualCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = STEPS_SUB_TERRAINS
        gen.num_rows = 10
        # mostly forward on stairs; the student's own ranges are wider
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.3, 0.5), (-0.2, 0.2), (-0.5, 0.5)


@configclass
class K1StepsResidualPlayEnvCfg(K1StepsResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        gen = self.scene.terrain.terrain_generator
        gen.num_rows, gen.num_cols, gen.curriculum = 5, 4, False
        self.scene.terrain.max_init_terrain_level = None
        self.events.push_robot = None
