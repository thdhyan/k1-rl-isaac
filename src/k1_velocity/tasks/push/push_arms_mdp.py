"""Push v4 (``Isaac-Push-Arms-K1-v0``): face-anchored absolute IK arms + arm-only push rewards.

Fixes the two failures seen in the push-simple / MAPPO videos (2026-10-12):

* Arms flail. The wrist terms were RELATIVE diff IK at 5 cm/step: every step of policy
  noise became a fresh random wrist delta, and a zero action did not hold the arm. Here
  each arm takes 2 numbers (lateral u, height v) in [-1, 1] that pick a point on the box
  face nearest the robot, pressed ``press_depth`` inside the surface. The target is
  clamped to the arm's reach sphere, EMA-smoothed, and tracked by ABSOLUTE DLS IK, so
  the same action holds the same tip position and noise moves the tip only along the face.
  The left arm owns the robot-left half of the face and the right arm the right half, so
  the stubs cannot cross.
* Robot pushes face-first with its trunk. ``hands_to_box`` measured the hand-link ORIGIN,
  which sits at the ELBOW (forearm mesh runs 0 -> 0.228 m along link +-y), so the policy
  was paid to bring elbows (and the chest behind them) to the box. Every term here uses
  the stub TIP (link origin + 0.20 m along the forearm). Box progress is paid only while
  both tips touch the box, and trunk/head closer than ``margin`` to the box is penalized.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import push_mdp as pmdp

#: stub tip in the hand-link frame (forearm mesh tip is at |y| = 0.228, 0.03 rounded cap)
STUB_TIP_LEFT = (0.0, 0.20, 0.0)
STUB_TIP_RIGHT = (0.0, -0.20, 0.0)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _box_pose_w(env):
    box = env.scene["box"]
    return box.data.root_pos_w.torch, box.data.root_quat_w.torch


def points_box_gap(env, pts_w: torch.Tensor) -> torch.Tensor:
    """(n,k) distance from k world points per env to the box surface (0 inside)."""
    pos, quat = _box_pose_w(env)
    half = pmdp._state(env).half_extents[:, None, :]
    p = pmdp._rot_batch(quat, pts_w - pos[:, None, :], inverse=True)
    return (p - torch.maximum(torch.minimum(p, half), -half)).norm(dim=-1)


def stub_tips_w(env, asset_name: str = "robot") -> torch.Tensor:
    """(n,2,3) left/right stub tips in the world frame."""
    robot = env.scene[asset_name]
    names = [n.split("/")[-1] for n in robot.body_names]
    idx = [names.index(pmdp.LEFT_EE), names.index(pmdp.RIGHT_EE)]
    off = torch.tensor([STUB_TIP_LEFT, STUB_TIP_RIGHT], device=env.device).expand(env.num_envs, 2, 3)
    pos = robot.data.body_pos_w.torch[:, idx, :]
    quat = robot.data.body_quat_w.torch[:, idx, :]
    return pos + math_utils.quat_apply(quat.reshape(-1, 4), off.reshape(-1, 3)).view(-1, 2, 3)


def face_target_base(env, side: float, u: torch.Tensor, v: torch.Tensor, press_depth: float,
                     edge_margin: float, asset_name: str = "robot") -> torch.Tensor:
    """(n,3) point on the box face nearest the robot, in the robot base frame.

    ``u`` in [-1, 1] spans this arm's half of the face (side +1 = robot-left half),
    ``v`` in [-1, 1] spans box centre height -> top minus edge_margin.
    """
    robot = env.scene[asset_name]
    root_pos, root_quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
    box_pos, box_quat = _box_pose_w(env)
    half = pmdp._state(env).half_extents
    # robot origin in the box frame picks the face (largest |d|/half over x, y)
    d = math_utils.quat_apply_inverse(box_quat, root_pos - box_pos)
    use_x = (d[:, 0] / half[:, 0]).abs() >= (d[:, 1] / half[:, 1]).abs()
    sgn = lambda x: torch.where(x >= 0, 1.0, -1.0)
    n = torch.zeros_like(d)
    n[:, 0] = torch.where(use_x, sgn(d[:, 0]), 0.0)
    n[:, 1] = torch.where(use_x, 0.0, sgn(d[:, 1]))
    half_n = torch.where(use_x, half[:, 0], half[:, 1])
    half_t = torch.where(use_x, half[:, 1], half[:, 0])
    z = torch.zeros_like(d)
    z[:, 2] = 1.0
    left = torch.linalg.cross(z, -n)                      # robot-left along the face
    lat = side * (0.1 + 0.4 * (u + 1.0)) * (half_t - edge_margin).clamp(min=0.0)   # 10-90 % of the half
    hgt = 0.5 * (v + 1.0) * (half[:, 2] - edge_margin).clamp(min=0.0)
    p_box = n * (half_n - press_depth)[:, None] + left * lat[:, None] + z * hgt[:, None]
    p_w = box_pos + math_utils.quat_apply(box_quat, p_box)
    return math_utils.quat_apply_inverse(root_quat, p_w - root_pos)


# ---------------------------------------------------------------------------
# action
# ---------------------------------------------------------------------------
class FaceAnchoredArmAction(DifferentialInverseKinematicsAction):
    """2-dim (u, v) per arm -> absolute IK target on the near box face (module docstring)."""

    cfg: "FaceAnchoredArmActionCfg"

    def __init__(self, cfg: "FaceAnchoredArmActionCfg", env):
        super().__init__(cfg, env)
        ids, names = self._asset.find_bodies(cfg.shoulder_body)
        if len(ids) != 1:
            raise ValueError(f"shoulder_body {cfg.shoulder_body!r} matched {names}")
        self._shoulder_idx = ids[0]
        self._target_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def target_b(self) -> torch.Tensor:
        return self._target_b

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = actions.clamp(-1.0, 1.0)
        self._processed_actions[:] = a
        ee_pos, ee_quat = self._compute_frame_pose()
        tgt = face_target_base(self._env, self.cfg.side, a[:, 0], a[:, 1], self.cfg.press_depth,
                               self.cfg.edge_margin, self.cfg.asset_name)
        # keep the target inside the reach sphere around the shoulder: when the box is
        # far the arm points at it at full extension instead of chasing an unreachable goal
        root_pos, root_quat = self._asset.data.root_pos_w.torch, self._asset.data.root_quat_w.torch
        sh = math_utils.quat_apply_inverse(
            root_quat, self._asset.data.body_pos_w.torch[:, self._shoulder_idx] - root_pos)
        r = (tgt - sh).norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # face behind/beside the shoulder or far away -> hold a forward ready pose instead of
        # splaying the arm sideways/backward toward it (2026-10-12 smoke frames)
        use_face = (tgt[:, :1] > sh[:, :1] + self.cfg.min_forward) & (r <= self.cfg.reach_radius + self.cfg.far_slack)
        rx, ry, rz = self.cfg.ready_offset
        ready = sh + torch.tensor((rx, self.cfg.side * ry, rz), device=self.device)
        tgt = torch.where(use_face, sh + (tgt - sh) * r.clamp(max=self.cfg.reach_radius) / r, ready)
        prev = torch.where(self._fresh[:, None], ee_pos, self._target_b)
        self._target_b[:] = self.cfg.smoothing * prev + (1.0 - self.cfg.smoothing) * tgt
        self._fresh[:] = False
        self._ik_controller.set_command(self._target_b, ee_pos, ee_quat)

    def _compute_frame_jacobian(self):
        """Root-frame Jacobian of the stub tip (position rows; rotation rows untouched).

        Isaac Lab's version adds skew(offset) with the offset still in the HAND frame to a
        ROOT-frame Jacobian. With the 0.20 m stub and an arbitrary hand orientation that
        sent the IK the wrong way (2026-10-12 probe_arm_ik.py), so rotate it first.
        """
        self._jacobian_b[:] = self.jacobian_b
        root_quat = self._asset.data.root_quat_w.torch
        body_quat = self._asset.data.body_quat_w.torch[:, self._body_idx]
        q_rb = math_utils.quat_mul(math_utils.quat_inv(root_quat), body_quat)
        r = math_utils.quat_apply(q_rb, self._offset_pos)
        self._jacobian_b[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(r), self._jacobian_b[:, 3:, :])
        return self._jacobian_b

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._fresh[slice(None) if env_ids is None else env_ids] = True


@configclass
class FaceAnchoredArmActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = FaceAnchoredArmAction
    side: float = 1.0              # +1 left arm (robot-left half of the face), -1 right
    shoulder_body: str = "Left_Arm_1"
    reach_radius: float = 0.40     # shoulder -> stub tip is ~0.46 m at full extension
    press_depth: float = 0.02      # target this far inside the face -> steady contact force
    edge_margin: float = 0.06
    smoothing: float = 0.8         # EMA per 50 Hz step (~0.1 s time constant)
    min_forward: float = 0.10      # face target must be this far ahead of the shoulder (base x)
    far_slack: float = 0.25        # ...and within reach_radius + far_slack, else ready pose
    # ready tip, shoulder-relative (base), y for the left arm: the carry hold pose at 0.325 m
    # width (carry_mdp.hold_pose), which is also the push-arms start pose -> no reset transient
    ready_offset: tuple[float, float, float] = (0.29, 0.086, -0.145)


# ---------------------------------------------------------------------------
# observations / rewards
# ---------------------------------------------------------------------------
def stub_tips_base(env, asset_name: str = "robot") -> torch.Tensor:
    """(n,6) stub tips in the robot base frame."""
    return pmdp._to_base(env, stub_tips_w(env, asset_name), asset_name=asset_name).reshape(env.num_envs, 6)


def stub_box_gap(env, asset_name: str = "robot") -> torch.Tensor:
    """(n,2) stub-tip distance to the box surface."""
    return points_box_gap(env, stub_tips_w(env, asset_name))


def stub_box_proximity(env, coarse: float = 0.30, fine: float = 0.05, asset_name: str = "robot") -> torch.Tensor:
    """Two-scale exp(-gap) on the FARTHER stub tip (0..2), like wrist_box_proximity."""
    g = stub_box_gap(env, asset_name).max(-1).values
    return torch.exp(-g / coarse) + torch.exp(-g / fine)


def body_box_contact_count(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """(n,) number of sensor bodies pressing on the box (filtered contact force > threshold N)."""
    f = env.scene.sensors[sensor_cfg.name].data.force_matrix_w
    f = f.torch if hasattr(f, "torch") else f                    # (n, bodies, filters, 3)
    return (f[:, sensor_cfg.body_ids].norm(dim=-1).amax(-1) > threshold).sum(-1).float()


def box_progress_arm_gated(env, contact_gap: float = 0.06, asset_name: str = "robot",
                           body_sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """box_goal_progress_rate, paid only while BOTH stub tips are within contact_gap of the box
    and (with ``body_sensor_cfg``) no non-arm body touches it.

    Always calls the rate func so its prev-distance baseline advances every step.
    """
    prog = pmdp.box_goal_progress_rate(env, asset_name)
    ok = stub_box_gap(env, asset_name).max(-1).values < contact_gap
    if body_sensor_cfg is not None:
        ok &= body_box_contact_count(env, body_sensor_cfg) == 0
    return prog * ok.float()


def box_progress_arms_proportional(env, contact_gap: float = 0.06, asset_name: str = "robot",
                                   body_sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """box_goal_progress_rate, scaled by the fraction of stub tips in contact (0 / 0.5 / 1.0)
    and vetoed (with ``body_sensor_cfg``) while a non-arm body touches the box.

    Push v5: v4's all-or-nothing gate never fired once in 3000 it (hands_to_box ~0.07,
    tips ~0.5 m away), so one hand on the box already earns half the progress.

    Always calls the rate func so its prev-distance baseline advances every step.
    """
    prog = pmdp.box_goal_progress_rate(env, asset_name)
    contact = (stub_box_gap(env, asset_name) < contact_gap).float().sum(-1) / 2.0
    if body_sensor_cfg is not None:
        contact = contact * (body_box_contact_count(env, body_sensor_cfg) == 0).float()
    return prog * contact


def body_box_intrusion(env, asset_cfg: SceneEntityCfg, margin: float = 0.18) -> torch.Tensor:
    """sum over bodies of clamp(margin - gap, 0) / margin (0 when clear, 1 per body at contact)."""
    robot = env.scene[asset_cfg.name]
    pts = robot.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    return ((margin - points_box_gap(env, pts)).clamp(min=0.0) / margin).sum(-1)


# ---------------------------------------------------------------------------
# Push v6 rewards (2026-10-12 user request): the v4/v5 audit showed an avoidance
# optimum - falls decayed but so did hands_to_box, and the box never moved. The
# fix set: a dense approach gradient from spawn, a pointing reward so directing
# the arms at the box pays even at range, and a strong tip-contact bonus big
# enough to offset the -10 body-contact/-5 intrusion/-500 fall terms.
# ---------------------------------------------------------------------------

def torso_to_box_exp(env, std: float = 0.6, asset_name: str = "robot") -> torch.Tensor:
    """exp(-d_xy / std) on trunk-to-box-centre distance: dense pull across the spawn
    annulus (0.5-1.8 m). The tip-only two-scale proximity has ~zero gradient beyond
    ~1 m (exp(-1.2/0.3) = 0.02), so from spawn the policy never learns to walk to the
    box. std 0.6 gives 0.44 at 0.5 m, 0.05 at 1.8 m."""
    robot = env.scene[asset_name]
    box_pos, _ = _box_pose_w(env)
    d = (robot.data.root_pos_w.torch[:, :2] - box_pos[:, :2]).norm(dim=-1)
    return torch.exp(-d / std)


def stub_box_pointing(env, asset_name: str = "robot") -> torch.Tensor:
    """mean over arms of relu(cos(shoulder->tip, tip->box)) in [0, 1].

    Dense gradient for directing the arms at the box even when contact is far away.
    relu so a backward arm is simply unpaid (0), never negative - negative rewards
    from a shaping term would trade against the contact terms unpredictably.
    """
    robot = env.scene[asset_name]
    names = [n.split("/")[-1] for n in robot.body_names]
    sh_idx = [names.index("Left_Arm_1"), names.index("Right_Arm_1")]
    sh = robot.data.body_pos_w.torch[:, sh_idx, :]
    tip = stub_tips_w(env, asset_name)
    box_pos, _ = _box_pose_w(env)
    fwd = tip - sh
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    to_box = box_pos[:, None, :] - tip
    to_box = to_box / to_box.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.relu((fwd * to_box).sum(-1)).mean(-1)


def stub_box_contact(env, contact_gap: float = 0.06, asset_name: str = "robot") -> torch.Tensor:
    """fraction of the two stub tips within contact_gap of the box surface (0..1).

    v6: contact is SUPER rewarded (+2.0) because v4/v5 confirmed the -10 contact /
    -5 intrusion / -500 fall terms make touching the box a net loss at every range;
    the policy found the avoidance optimum. 2.0 x dt x both-tips = 0.08/step, more
    than the fall risk premium once standing is stable and -10 contact applies to
    non-arm bodies only, so tips are exempt."""
    return (stub_box_gap(env, asset_name) < contact_gap).float().mean(-1)


def stand_still_far(env, far_gap: float = 0.7, vel_thresh: float = 0.05,
                    asset_name: str = "robot") -> torch.Tensor:
    """1.0 while the base is nearly still AND the box is farther than far_gap.

    v6: the v5 final video shows a stand-and-wait policy (time_out 71%, falls 37%
    and falling). This term makes waiting strictly worse than approaching; it does
    not fire near the box, so a robot standing while pushing is not charged."""
    robot = env.scene[asset_name]
    box_pos, _ = _box_pose_w(env)
    d = (robot.data.root_pos_w.torch[:, :2] - box_pos[:, :2]).norm(dim=-1)
    v = robot.data.root_lin_vel_w.torch[:, :2].norm(dim=-1)
    return ((v < vel_thresh) & (d > far_gap)).float()
