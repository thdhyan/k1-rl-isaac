"""K1 embodiment for COMPASS (X-Mobility base + residual RL navigation specialist).

COMPASS drives every embodiment through a 6-dim velocity action, of which the locomotion
action reads [vx, vy, wz] = action[:, [0, 1, 5]]. For K1 the locomotion layer is the frozen
P2 walk student (models/k1_velocity_student_gaitfix.pt, 500 -> 12 legs, 10-step history,
50 Hz). The arms and head stay at their default pose.

Rates: COMPASS runs the env at 10 Hz (sim dt 0.005, decimation 20) and calls apply_actions
every physics step. The student was trained at 50 Hz, so it runs on every 4th physics step
and keeps its own gait clock (a counter advanced per student step, step_dt 0.02), the same
way push_mdp gives extra robots their own clock through _ClockView.

Launch through compass_k1/launch.py, which adds 'k1' to EmbodimentEnvCfgMap in run.py.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from k1_velocity.tasks.push.push_mdp import K1_LEG_JOINTS
from k1_velocity.tasks.skills.residual_action import HISTORY, load_student, stack_term_major
from k1_velocity.tasks.velocity import gait_clock

from mobility_es.config import scene_assets
from mobility_es.config.env_cfg import GoalReachingEnvCfg
from mobility_es.mdp.action.action_visualization import ActionVisualizer

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_PATH = os.environ.get("K1_COMPASS_STUDENT", os.path.join(_WS, "models/k1_velocity_student_gaitfix.pt"))
STUDENT_PERIOD = 4                      # physics steps per student step: 200 Hz / 50 Hz
VEL_LIMITS = (0.5, 0.3, 0.8)            # student command ranges (vx, vy, wz)
FALL_BODIES = ["Trunk"]


class _StudentClock:
    """Own gait-clock identity running at the student rate, not the 10 Hz env rate."""

    def __init__(self, num_envs: int, device):
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.num_envs = num_envs
        self.step_dt = 0.02


class K1StudentLocomotionAction(ActionTerm):
    """6-dim COMPASS velocity action -> frozen K1 walk student -> 12 leg position targets."""

    cfg: "K1StudentLocomotionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        dev = self.device
        self._policy = load_student(cfg.student_path, dev)
        leg_ids, _ = self._asset.find_joints(K1_LEG_JOINTS, preserve_order=False)  # asset order, as push_mdp
        self._leg_ids = torch.as_tensor(leg_ids, dtype=torch.long, device=dev)
        self._default12 = self._asset.data.default_joint_pos[:, self._leg_ids].clone()
        self._a_cfg = SceneEntityCfg(cfg.asset_name)
        self._a_cfg.resolve(env.scene)
        self._raw = torch.zeros(self.num_envs, 6, device=dev)
        self._vel = torch.zeros(self.num_envs, 3, device=dev)
        self._last = torch.zeros(self.num_envs, 12, device=dev)
        self._hist = torch.zeros(self.num_envs, HISTORY, 50, device=dev)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
        self._targets = self._default12.clone()
        self._clock = _StudentClock(self.num_envs, dev)
        self._substep = 0
        self._lim = torch.tensor(VEL_LIMITS, device=dev)
        self._visualizer = ActionVisualizer(cfg, env)
        print(f"[compass-k1] walk student loaded: {cfg.student_path} (500 -> 12, every "
              f"{STUDENT_PERIOD} physics steps)")

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._vel = torch.maximum(torch.minimum(actions[:, [0, 1, 5]], self._lim), -self._lim)
        self._substep = 0

    def _student_step(self):
        env = self._env
        lin = mdp.base_lin_vel(env, asset_cfg=self._a_cfg)
        ang = mdp.base_ang_vel(env, asset_cfg=self._a_cfg)
        grav = mdp.projected_gravity(env, asset_cfg=self._a_cfg)
        robot = self._asset
        leg_pos = robot.data.joint_pos[:, self._leg_ids] - robot.data.default_joint_pos[:, self._leg_ids]
        leg_vel = robot.data.joint_vel[:, self._leg_ids]
        self._clock.episode_length_buf += 1
        clock = gait_clock.phase_clock_from_magnitude(self._clock, self._vel[:, :2].norm(dim=-1))
        frame = torch.cat([lin, ang, grav, self._vel, leg_pos, leg_vel, self._last, clock], dim=-1)
        self._hist = torch.roll(self._hist, shifts=-1, dims=1)
        self._hist[:, -1] = frame
        if self._fresh.any():
            self._hist[self._fresh] = frame[self._fresh].unsqueeze(1).expand(-1, HISTORY, -1)
            self._fresh[:] = False
        with torch.inference_mode():
            out = self._policy(stack_term_major(self._hist))
        self._last = out.clone()
        self._targets = self._default12 + out.clamp(-1.0, 1.0)

    def apply_actions(self):
        if self._substep % STUDENT_PERIOD == 0:
            self._student_step()
        self._substep += 1
        self._asset.set_joint_position_target(self._targets, joint_ids=self._leg_ids.to(torch.int32))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._last[env_ids] = 0.0
        self._fresh[env_ids] = True
        self._clock.episode_length_buf[env_ids] = 0
        self._targets[env_ids] = self._default12[env_ids]

    def visualize(self, base_action, residual_action):
        self._visualizer.visualize(base_action, residual_action)


@configclass
class K1StudentLocomotionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = K1StudentLocomotionAction
    student_path: str = STUDENT_PATH


@configclass
class K1ActionsCfg:
    drive_joints = K1StudentLocomotionActionCfg(asset_name="robot")


@configclass
class K1GoalReachingEnvCfg(GoalReachingEnvCfg):
    """K1 reaching a 2D goal pose with the walk student as locomotion layer."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Head pitch link; same mount as the K1 head-camera tasks (head_env_cfg.py).
        self.scene.camera = scene_assets.camera.replace(prim_path="{ENV_REGEX_NS}/Robot/Head_2/front_cam")
        self.scene.camera.offset = CameraCfg.OffsetCfg(pos=(0.10, 0.0, 0.10),
                                                       rot=(0.5, -0.5, 0.5, -0.5),
                                                       convention="ros")
        self.actions = K1ActionsCfg()
        fall = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FALL_BODIES), "threshold": 1.0}
        self.observations.eval.fall_down = ObsTerm(func=mdp.illegal_contact, params=fall)
        self.terminations.base_contact = DoneTerm(func=mdp.illegal_contact, params=fall)
        self.events.base_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                    "force_range": (0.0, 0.0), "torque_range": (0.0, 0.0)},
        )
        # Student stands only with ground friction >= 0.7 (push task finding); COMPASS uses 1.0.
