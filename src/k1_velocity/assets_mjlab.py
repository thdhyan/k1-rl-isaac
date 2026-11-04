"""K1 as trained in mjlab (``mjlab_k1``), for Isaac Lab: soft Booster gains from the exported USD.

USD source: ``mjlab_k1/scripts_export_mjcf.py`` writes ``K1_22dof_mjlab.xml`` (the mjlab-compiled model:
Booster 4 Hz PD gains, armature, feet-only collision, home keyframe), then Isaac Lab's
``scripts/tools/convert_mjcf.py`` turns it into ``usd_mjlab/K1_22dof_mjlab/K1_22dof_mjlab.usda``.

Gains here repeat the mjlab recipe (kp = armature * (2 pi 4 Hz)^2, kd = 2 zeta armature (2 pi 4 Hz),
ankle armature ratio 1.4 / 0.4) so the Isaac drives equal the USD's and mjlab's. Implicit actuators: the
PD runs inside the solver as in mjlab's builtin position actuators. mjlab also delays commands by 2-8
physics steps; that delay is not modelled here (policies are trained robust to it).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_USD_DIR = Path(__file__).resolve().parents[4] / "src/k1_description/assets/robots/K1/usd_mjlab"
K1_MJLAB_USD = os.environ.get("K1_MJLAB_USD", str(_USD_DIR / "K1_22dof_mjlab/K1_22dof_mjlab.usda"))
#: same model plus Booster's arm/hand collision meshes (``scripts_export_mjcf.py --arms``), for push/carry
K1_MJLAB_ARMS_USD = os.environ.get(
    "K1_MJLAB_ARMS_USD", str(_USD_DIR / "K1_22dof_mjlab_arms/K1_22dof_mjlab_arms.usda"))

#: canonical leg order = mjlab model order (also the deploy / ROS order)
LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
UPPER_REGEX = [".*Head.*", ".*Shoulder.*", ".*Elbow.*"]

_W = 4.0 * 2.0 * math.pi
# (joint regex, effort, armature, zeta)
_JOINTS = (
    (".*_Hip_Pitch", 68.0, 0.0478125, 1.5),
    (".*_Hip_Roll", 76.0, 0.0339552, 1.5),
    (".*_Hip_Yaw", 38.3, 0.0282528, 1.5),
    (".*_Knee_Pitch", 112.0, 0.095625, 1.0),
    (".*_Ankle_Pitch", 38.3, 0.0282528 * 1.4, 1.5),
    (".*_Ankle_Roll", 38.3, 0.0282528 * 0.4, 1.5),
)
_UPPER = (((".*Shoulder.*", ".*Elbow.*"), 14.0), ((".*Head.*",), 6.0))


def _group(names, effort, armature, zeta) -> ImplicitActuatorCfg:
  return ImplicitActuatorCfg(
    joint_names_expr=list(names), stiffness=armature * _W**2, damping=2.0 * zeta * armature * _W,
    armature=armature, effort_limit_sim=effort,
    # all explicit: a None falls back to the USD value, which Newton returns as (1, n) when
    # replicate_physics=False (push/carry) and ActuatorBase rejects it. mjlab: no joint friction or speed limit.
    velocity_limit_sim=100.0, friction=0.0, dynamic_friction=0.0, viscous_friction=0.0)


_ACTUATORS = {n.strip(".*_").lower(): _group((n,), e, a, z) for n, e, a, z in _JOINTS}
_ACTUATORS["arms"] = _group(_UPPER[0][0], _UPPER[0][1], 0.001, 1.0)
_ACTUATORS["head"] = _group(_UPPER[1][0], _UPPER[1][1], 0.001, 1.0)

#: action = default + scale * a, with |a| = 1 -> 0.25 * effort / kp (mjlab K1_ACTION_SCALE)
K1_MJLAB_ACTION_SCALE = {n: 0.25 * e / (a * _W**2) for n, e, a, _ in _JOINTS}

K1_MJLAB_CFG = ArticulationCfg(
  spawn=sim_utils.UsdFileCfg(
    usd_path=K1_MJLAB_USD,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
  ),
  # mjlab HOME: knees bent, flat feet, trunk 0.56 m
  init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.56),
    joint_pos={
      ".*_Hip_Pitch": -0.3, ".*_Knee_Pitch": 0.6, ".*_Ankle_Pitch": -0.3,
      "Left_Shoulder_Roll": -1.3, "Right_Shoulder_Roll": 1.3,
    },
    joint_vel={".*": 0.0},
  ),
  soft_joint_pos_limit_factor=0.9,
  actuators=_ACTUATORS,
)
K1_MJLAB_ARMS_CFG = K1_MJLAB_CFG.replace(spawn=K1_MJLAB_CFG.spawn.replace(usd_path=K1_MJLAB_ARMS_USD))


#: head pitch link under the robot prim: the MJCF importer nests bodies along the kinematic tree, the URDF one doesn't
K1_HEAD_LINK = "Head_2" if os.environ.get("K1_ASSET", "mjlab") == "urdf" else "Geometry/Trunk/Head_1/Head_2"


def k1_robot_cfg(arms: bool = False) -> ArticulationCfg:
  """Robot for the Isaac K1 tasks. ``K1_ASSET=mjlab`` (default): soft-gain mjlab USD; ``K1_ASSET=urdf``:
  booster_train's URDF with delayed Booster PD (the asset the pre-2026-10-28 Isaac checkpoints used)."""
  if os.environ.get("K1_ASSET", "mjlab") == "urdf":
    from booster_train.assets.robots.booster import BOOSTER_K1_CFG
    return BOOSTER_K1_CFG.copy()
  return (K1_MJLAB_ARMS_CFG if arms else K1_MJLAB_CFG).copy()
