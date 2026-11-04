"""Push v4: arm-only pushing with face-anchored absolute IK (``Isaac-Push-Arms-K1-v0``).

Same world, curricula and terminations as push simple. What changes (push_arms_mdp.py):

* Action 8 = [vx, vy, wz, H*] frozen base | (u, v) left arm | (u, v) right arm.
* Rewards measure the stub TIPS, not the hand-link origin (= elbow); box progress is
  paid only with both tips on the box; trunk/head near the box is penalized; arm joint
  velocity and action rate are penalized so the arms move smoothly.

Base: P2 walk student (H* ignored) until the Stage-0 lower-body base is exported.
"""
from __future__ import annotations

import copy

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from ..carry import carry_mdp
from . import push_arms_mdp as amdp
from . import push_env_cfg as base
from . import push_mdp as mdp
from .push_simple_env_cfg import K1PushSimpleEnvCfg, K1PushSimplePlayEnvCfg

#: arm PD gains as in carry (default kp 3.9 / kd 0.25 sagged 14-18 cm short of the IK target)
ARM_STIFFNESS = 40.0
ARM_DAMPING = 2.0

#: every robot body except the arm links (Left/Right_Arm_*, *_hand_link) - these must not touch the box
_BODY_ON_BOX = SceneEntityCfg("box_contact", body_names=["^(?!.*(Arm|hand)).*$"])

_ARMS = SceneEntityCfg("robot", joint_names=list(mdp.K1_LEFT_ARM_JOINTS) + list(mdp.K1_RIGHT_ARM_JOINTS))


def _arm(side: str) -> amdp.FaceAnchoredArmActionCfg:
    left = side == "left"
    return amdp.FaceAnchoredArmActionCfg(
        asset_name="robot",
        joint_names=mdp.K1_LEFT_ARM_JOINTS if left else mdp.K1_RIGHT_ARM_JOINTS,
        body_name=mdp.LEFT_EE if left else mdp.RIGHT_EE,
        body_offset=amdp.FaceAnchoredArmActionCfg.OffsetCfg(
            pos=amdp.STUB_TIP_LEFT if left else amdp.STUB_TIP_RIGHT),
        scale=1.0,
        side=1.0 if left else -1.0,
        shoulder_body="Left_Arm_1" if left else "Right_Arm_1",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
    )


def _box_contact_sensor(env_cfg) -> None:
    """Robot-body contact forces filtered to the box (body_box_contact, box_progress gate).

    The box prim is renamed: PhysX's ``env_*`` glob also crosses '/', so ``env_*/box``
    matched the robot's own collision prims named ``box`` (Trunk, both shanks) too.
    """
    env_cfg.scene.box = env_cfg.scene.box.replace(prim_path="{ENV_REGEX_NS}/PushBox")
    env_cfg.scene.box_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1, filter_prim_paths_expr=["{ENV_REGEX_NS}/PushBox"])


def _stiff_arms(env_cfg) -> None:
    """Arm PD gains -> carry values and arms start in the carry hold pose (bent elbows),
    on a private copy of the robot cfg. From the default straight-down arm the first DLS
    steps snapped the elbow (Elbow_Yaw) to full flex and IK stalled 14 cm short."""
    robot = copy.deepcopy(env_cfg.scene.robot)
    robot.actuators["arms"].stiffness = ARM_STIFFNESS
    robot.actuators["arms"].damping = ARM_DAMPING
    q_l, q_r = carry_mdp.hold_pose(0.325)
    joint_pos = dict(robot.init_state.joint_pos)
    joint_pos.update({n: float(v) for n, v in zip(carry_mdp.ARM_JOINTS, list(q_l) + list(q_r))})
    robot.init_state.joint_pos = joint_pos
    env_cfg.scene.robot = robot


@configclass
class K1PushArmsActionsCfg:
    base_velocity = mdp.FrozenBaseVelocityActionCfg(asset_name="robot", base_policy_path=base.WALK_STUDENT_POLICY)
    arm_left = _arm("left")
    arm_right = _arm("right")


@configclass
class K1PushArmsObservationsCfg(base.ObservationsCfg):
    @configclass
    class TeacherCfg(base.ObservationsCfg.TeacherCfg):
        stub_tips = ObsTerm(func=amdp.stub_tips_base)     # 6

    teacher: TeacherCfg = TeacherCfg()


@configclass
class K1PushArmsRewardsCfg:
    # 1. reach: stub TIPS to the box (farther tip, two-scale, 0..2)
    hands_to_box = RewTerm(func=amdp.stub_box_proximity, weight=1.0)
    # 2. box closer to the goal - progress only while both tips touch the box
    box_progress = RewTerm(func=amdp.box_progress_arm_gated, weight=5.0, params={
        "contact_gap": 0.06, "body_sensor_cfg": _BODY_ON_BOX})
    box_near_goal = RewTerm(func=mdp.box_goal_near, weight=2.0, params={"std": 0.5})
    success_bonus = RewTerm(func=mdp.success_bonus, weight=500.0)
    # 3. push with the arms, not the body
    body_box_intrusion = RewTerm(func=amdp.body_box_intrusion, weight=-5.0, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["Trunk", "Head_2"]), "margin": 0.18})
    body_box_contact = RewTerm(func=amdp.body_box_contact_count, weight=-10.0, params={"sensor_cfg": _BODY_ON_BOX})
    flat_orientation_l2 = RewTerm(func=mdp.vmdp.flat_orientation_l2, weight=-2.0)
    # 4. do not fall
    termination_penalty = RewTerm(func=mdp.term_fired, weight=-500.0,
                                  params={"term_names": ["fallen", "box_oob", "box_tip"]})
    # 5. smooth arms
    action_rate_l2 = RewTerm(func=mdp.vmdp.action_rate_l2, weight=-0.01)
    arm_joint_vel_l2 = RewTerm(func=mdp.vmdp.joint_vel_l2, weight=-2e-3, params={"asset_cfg": _ARMS})
    joint_pos_limits = RewTerm(func=mdp.vmdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": _ARMS})


@configclass
class K1PushArmsEnvCfg(K1PushSimpleEnvCfg):
    actions: K1PushArmsActionsCfg = K1PushArmsActionsCfg()
    observations: K1PushArmsObservationsCfg = K1PushArmsObservationsCfg()
    rewards: K1PushArmsRewardsCfg = K1PushArmsRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.base_velocity.base_policy_path = base.WALK_STUDENT_POLICY
        _stiff_arms(self)
        _box_contact_sensor(self)


@configclass
class K1PushArmsPlayEnvCfg(K1PushSimplePlayEnvCfg):
    actions: K1PushArmsActionsCfg = K1PushArmsActionsCfg()
    observations: K1PushArmsObservationsCfg = K1PushArmsObservationsCfg()
    rewards: K1PushArmsRewardsCfg = K1PushArmsRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.teacher.enable_corruption = False
        _stiff_arms(self)
        _box_contact_sensor(self)


# ---------------------------------------------------------------------------
# Push v5 (``Isaac-Push-ArmsV5-K1-v0``): same world/actions/obs as v4, three
# reward changes from the v4 audit (box_progress stayed 0.0 for all 3000 it):
# one hand on the box earns half the progress, box_near_goal dropped (paid
# 0.18/s while the box never moved), arm-velocity penalty 2e-3 -> 5e-4 (was
# larger than hands_to_box). Still on the OLD walk base (no H*); whether the
# new Stage-0 student is needed is judged from the v5 video + numbers.
# ---------------------------------------------------------------------------
@configclass
class K1PushArmsV5RewardsCfg(K1PushArmsRewardsCfg):
    box_progress = RewTerm(func=amdp.box_progress_arms_proportional, weight=5.0, params={
        "contact_gap": 0.06, "body_sensor_cfg": _BODY_ON_BOX})
    box_near_goal = None
    arm_joint_vel_l2 = RewTerm(func=mdp.vmdp.joint_vel_l2, weight=-5e-4, params={"asset_cfg": _ARMS})


@configclass
class K1PushArmsV5EnvCfg(K1PushArmsEnvCfg):
    rewards: K1PushArmsV5RewardsCfg = K1PushArmsV5RewardsCfg()


@configclass
class K1PushArmsV5PlayEnvCfg(K1PushArmsPlayEnvCfg):
    rewards: K1PushArmsV5RewardsCfg = K1PushArmsV5RewardsCfg()


# ---------------------------------------------------------------------------
# Push v6 (``Isaac-Push-ArmsV6-K1-v0``): v5 rewards + dense approach / pointing /
# super contact reward / stand-still penalty, and the box spawns super close
# (0.5-0.65 m) early so contact happens by accident while the approach gradient
# is learned. The v5 run never touched the box once: falls declined and the
# policy settled into an avoidance optimum.
# ---------------------------------------------------------------------------
@configclass
class K1PushArmsV6RewardsCfg(K1PushArmsV5RewardsCfg):
    torso_to_box = RewTerm(func=amdp.torso_to_box_exp, weight=1.0, params={"std": 0.6})
    stub_pointing = RewTerm(func=amdp.stub_box_pointing, weight=1.0)
    stub_contact = RewTerm(func=amdp.stub_box_contact, weight=2.0, params={"contact_gap": 0.06})
    stand_still_far = RewTerm(func=amdp.stand_still_far, weight=-0.5, params={"far_gap": 0.7})


@configclass
class K1PushArmsV6EnvCfg(K1PushArmsV5EnvCfg):
    rewards: K1PushArmsV6RewardsCfg = K1PushArmsV6RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # super-close spawns early: near face ~0.35 m, at the frozen base's stop
        # radius, so an accidental lean touches the box
        self.curriculum.box_dist.params["r0"] = (0.5, 0.65)


@configclass
class K1PushArmsV6PlayEnvCfg(K1PushArmsV5PlayEnvCfg):
    rewards: K1PushArmsV6RewardsCfg = K1PushArmsV6RewardsCfg()
