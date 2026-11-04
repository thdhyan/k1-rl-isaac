# K1 Head Tracking (P3)

**Status: implemented** — the live task lives in the velocity package:

- Env cfg: [`isaac_tasks/k1_velocity/source/k1_velocity/tasks/head/head_env_cfg.py`](../k1_velocity/source/k1_velocity/tasks/head/head_env_cfg.py)
- MDP (detector, rewards, ball-speed curriculum, CCW search): [`head_mdp.py`](../k1_velocity/source/k1_velocity/tasks/head/head_mdp.py)
- Gym id: **`Isaac-HeadTrack-K1-v0`** · wandb `p3_head_track`
- Launcher: `scripts/spark_soccer_host.sh p3` (smoke-gated 16×3 → 512×2000)

The old draft package (`source/`) was removed: it used a fake velocity-command
"ball target" with GT angles, which contradicts the P3 spec (deployable obs =
camera detections only, no ground truth).

## Task

Keep the **detected** ball centred in the head camera. Head motors only
(`AAHead_yaw`, `Head_pitch`); legs are held at the standing default — tracking is
independent of locomotion and balance.

## Obs (12, no ground truth)

| term | dim | source |
|---|---|---|
| detection | 3 | `(visible, du, dv)` — head-cam RGB @10 Hz → YOLOv8n `sports ball` bbox centre, held between detections (geometric FOV proxy only if the detector is unavailable) |
| head joint pos | 2 | `AAHead_yaw`, `Head_pitch` (+noise) |
| head joint vel | 2 | same (+noise) |
| base ang vel | 3 | +noise |
| last action | 2 | head action history |

## Action (2)

Head joint position targets, scale 0.5, default-offset.

## Rewards

| term | weight | signal |
|---|---|---|
| `ball_centered` | +2.0 | `exp(-(du²+dv²)/0.35²)` on the **detection** — the user's "keep the bounding box centred" reward |
| `track_ball_angle` | +1.0 | geometric head-pointing (GT, reward-time shaping only) |
| `ball_in_frame` | +0.5 | detector sees the ball |
| `action_rate_l2` | −0.1 | smoothness |
| `joint_pos_limits` | −1.0 | head limits |
| `time_penalty` | −0.01 | step count |

## Curriculum — ball speed

`ball_speed`: **0 → 0.8 m/s**, linear over iterations 200 → 1200. Static ball
first (centre the bounding box), then a slowly rolling ball the head must keep
tracked. Direction is resampled per reset; an interval event re-asserts the
speed against ground friction.

## CCW search (conditional)

`head_mdp.ccw_search_command` and the compose runtime
(`src/k1_sim_isaac/scripts/k1_soccer_compose.py::update_search`): when the
camera has no ball for 0.5 s, the locomotion policy receives an **in-place
counter-clockwise** command `(0, 0, +0.6 rad/s)` — rotate without x/y drift to
re-acquire an out-of-FOV ball.

---
> **Network note (2026-10-27):** the robotics router (192.168.50.1) has no internet uplink. The laptop shares its eduroam (`wlo1`) internet to the robots by NAT: `enp2s0` = 192.168.50.250/24 is the robots' gateway, with `ip_forward=1` and nft table `ip k1nat` (masquerade out `wlo1`, source 192.168.50.0/24). It is lost on laptop reboot; robots use DNS 8.8.8.8/1.1.1.1 (router DNS is wrong). Re-create steps: `docs/network_robotics_5G_2026-10-26.md` ("Robot internet via laptop").
