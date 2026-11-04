# jobs

Part of [k1-rl-isaac](https://github.com/thdhyan/k1-rl-isaac).

| File | What it does |
|---|---|
| `dl_ab_push_home_host.sh` | Host launcher (dl, GPU 2): Track B A/B hunt step A (HANDOFF "in-env A/B hunt", |
| `dl_eval_push_host.sh` | P6 metrics eval host wrapper (dl, GPU 2 = pinned record/eval GPU). |
| `dl_partial800_record_host.sh` | Host launcher (dl, GPU 2): render the gait-v2 base PROGRESS VIDEO at the last |
| `dl_partial800_walk_record_host.sh` | Host launcher (dl, GPU 2): WALKING variant of the gait-v2 base progress |
| `dl_push_base_resume_container.sh` | Track B gait-v2 base RESUME (container, runs INSIDE the isaac-lab image on |
| `eval_push.sh` | P6 metrics eval (inside the Isaac image). |
| `export_base_policy.sh` | Export the frozen base policy (latest partial-control ckpt) to TorchScript |
| `kb_lab_local.sh` | Local keyboard-control lab for the Booster K1 teacher policy (laptop, DISPLAY=:0). |
| `p3_recover_container.sh` | Recover P3's clean iteration-1900 checkpoint to the required iteration 1999. |
| `panel_smoke_container.sh` | Smoke-test the tiled debug panel recorder for any Track A task. |
| `play_partial_diag.sh` | DIAG (inside the Isaac image): play a partial-control checkpoint in its OWN |
| `probe_rewards_container.sh` |  |
| `push_preflight.sh` | ============================================================================= |
| `queue_student.sh` | queue_student.sh — server-side student launcher (runs inside tmux): |
| `record_p2_final_container.sh` | Record final gait-v2 P2 teacher + deployable student after their final marker. |
| `record_p2_one_container.sh` |  |
| `record_policies_container.sh` | REC (container-side): play + record finished checkpoints with play_record.py |
| `record_push.sh` | P6 play + RECORD (inside the Isaac image): video of a push/reach ckpt. |
| `reward_probe.sh` | Reward-term probe wrapper (inside the Isaac image): builds each task, runs |
| `sim2sim_mjlab_isaac.sh` | Inside the Isaac Lab 3.0 image: run an mjlab K1 checkpoint in Isaac Lab and record it. |
| `smoke_isaac_tasks.sh` | Inside the Isaac Lab 3.0 image (repo at /ws): build + zero-action step each K1 task, one process per task. |
| `smoke_p3_head.sh` | P3 head-track SMOKE ONLY (16x3, cameras + YOLO) — used to validate the env |
| `smoke_p4_teacher.sh` | P4 kick TEACHER SMOKE ONLY (16x3). Run on a spark inside the Isaac image. |
| `smoke_push.sh` | Box-push SMOKE (16x3, video) inside the Isaac image. Usage: smoke_push.sh reach/push |
| `spark_force_container.sh` | F (container-side): P1f/P2f force-variant trainings, SMOKE-gated and CHAINED: |
| `spark_full_container.sh` | FULL (container-side): full P1/P2 student runs on spark04 — the exact queue_student.sh |
| `spark_p2_dispcheck_container.sh` | Does the P2 policy actually translate, or does it just stand there? |
| `spark_p2_gait_container.sh` | P2 gait-v2 campaign: revised command distribution + H1/G1-style shaping. |
| `spark_p2_gaitfix_preflight_container.sh` | P2 gait-structure campaign, stage 1 of 2: preflight only. |
| `spark_p2_gaitfix_train_container.sh` | P2 gait-structure campaign, stage 2 of 2: smoke, full training, then BOTH gates. |
| `spark_p2_render_container.sh` | Render the delivery video for the verified P2 checkpoint. |
| `spark_p2_velcurriculum_container.sh` | P2 velocity-curriculum campaign. |
| `spark_p2_velresume_container.sh` | Verify a P2 checkpoint by eye before trusting it, then resume the velocity |
| `spark_partial_container.sh` | HP (hierarchical partial-control) container-side launcher: runs INSIDE |
| `spark_probe4.sh` | Quick batch probe (NO Kit boot, ~60 s): install all four of our extensions, |
| `spark_probe5.sh` | Probe5 v2: build an InteractiveScene exactly like training (robot + ground + |
| `spark_probe6.sh` | Probe6: distinguish WHY the image contact sensor reports only ('Trunk',): |
| `spark_push_base_container.sh` | Track B frozen-base retrain (container side, runs INSIDE |
| `spark_push_container.sh` | Push container launcher: WHICH=reach (stage 1) / push (stage 2, warm-start). |
| `spark_s1_container.sh` | S1 (container-side): install our booster_ws stack into the isaac-lab image and run |
| `spark_s2_container.sh` | S2 (container-side): install our booster_ws stack into the isaac-lab image and run |
| `spark_soccer_container.sh` | Soccer container launcher: WHICH=p3 (head track) / p4t (kick teacher). |
| `sweep_standing_container.sh` |  |
| `zzbw_push_host.sh` | ============================================================================= |
