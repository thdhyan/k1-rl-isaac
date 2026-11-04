# scripts

Part of [k1-rl-isaac](https://github.com/thdhyan/k1-rl-isaac).

| File | What it does |
|---|---|
| `eval_push_check.py` | P6 checkpoint metrics evaluator: run a trained push/reach policy headlessly and report box position error, contact success rate, wrist track |
| `flatten_k1_usd.py` | Flatten K1 USD hierarchy using Sdf layer manipulation. |
| `gait_gate.py` | Thin wrapper over isaac_tasks/k1_velocity/scripts/gait_gate.py (reused, not copied). |
| `keyboard_cmd.py` | Keyboard velocity commands for the interactive play scripts. |
| `make_zero_actor.py` | Save an UNTRAINED actor for the push task, so the environment can be run and rendered now. |
| `panel_video.py` | Small tiled debug-video helper for K1 play/eval runs. |
| `plateau_stop.py` | Reward-plateau early stopping for a running RSL-RL log. |
| `play.py` | Play (eval) a trained K1 velocity policy and optionally export it. |
| `play_contact_visualize.py` | Play K1 velocity student with contact sensor visualization. |
| `play_keyboard.py` | Keyboard-controlled play script for K1 student policy. |
| `play_keyboard_fixed.py` | Keyboard-controlled play script for K1 student policy. |
| `play_keyboard_teacher.py` | Keyboard teleop for a TEACHER (PPO) velocity checkpoint. |
| `play_mjlab_policy.py` | Sim2sim: run an mjlab-trained K1 velocity checkpoint in Isaac Lab (Newton or PhysX) and record it. |
| `play_push_ik.py` | Closed-loop Mjlab-PushIK checkpoint in Isaac Lab (K1_PHYSICS=physx/newton): the deploy controller drives two K1s and one box, physics decide |
| `play_record.py` | Play + RECORD a trained K1 policy headlessly: mp4 with a live HUD + .npz trace. |
| `play_student.py` | Play (eval) the distilled K1 velocity STUDENT policy and export it. |
| `ppo_guard.py` | Keep one bad PPO update from permanently poisoning the policy weights. |
| `probe_arm_ik.py` | Arm IK probe for push v4 (Isaac-Push-Arms-K1-*): does the stub tip reach the IK target? |
| `probe_k1_bodies.py` | Probe: spawn K1, dump articulation body/joint names + contact sensor bodies. |
| `probe_lowerbody_yaw.py` | Yaw-spin probe for the Stage-0 lower-body env (PLAN_HRL_CONTROLLER.md 2b audit). |
| `probe_rewards.py` | Reward probe: construct the P2 teacher env and exercise every reward term. |
| `probe_scene_resolver.py` | Probe: why does the push scene build fail? |
| `replay_kick_rtx.py` | Isaac Sim RTX replay of episodes recorded by mjlab_k1/scripts_kick_record.py: chase camera beside head camera. No physics runs here. Each li |
| `reward_probe_check.py` | Per-term REWARD probe: build an env, step N times with random actions, and print the mean RAW value of every reward term + contact-sensor di |
| `smoke_task.py` | Build one K1 Isaac Lab task, step it with zero actions, report shapes and trunk height. |
| `sweep_standing_gains.py` | Standing sweep: what leg stiffness lets the K1 hold itself up? |
| `task_docs.py` | Generate per-task REWARDS.md + ARCHITECTURE.png in each task folder. |
| `train.py` | Train K1 velocity locomotion policy with RSL-RL. |
| `train_kick.py` | Train K1 kick policy with RSL-RL. |
| `train_kick_student.py` | Train the K1 kick STUDENT policy via teacher-student distillation. |
| `train_mappo.py` | Train multi-robot push with skrl MAPPO: one policy PER robot + centralized critics. |
| `train_student.py` | Train the K1 velocity STUDENT policy via teacher-student distillation. |
| `zero_step_check.py` | ZERO-AGENT single-step verifier: build an env, reset, run N steps with zero actions, and prove one full step completes (obs -> act -> physic |
