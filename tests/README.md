# tests

Part of [k1-rl-isaac](https://github.com/thdhyan/k1-rl-isaac).

| File | What it does |
|---|---|
| `test_agile_parity.py` | Tests for AGILE T1 reward parity. |
| `test_carry_decoupled_policy.py` | Carry actor: two independent MLPs (12 legs, 8 arms) behind rsl_rl's MLPModel API. |
| `test_carry_tiers.py` | Tiered carry rewards: the pure-tensor pieces in carry_tiers, plus cfg wiring. |
| `test_deployability.py` | Static deployability test (PLAN_PHASE6_SOCCER_HRL / TASKS T6.3.5). |
| `test_distill_distribution_cfg.py` | The distillation student AND teacher must both be stochastic. |
| `test_env_cfg_wiring.py` | Static audit that every Isaac Lab manager group is actually loadable. |
| `test_feet_clearance_lift.py` | `feet_clearance` must measure LIFT, not absolute foot height. |
| `test_gait_rewards.py` | Tests for the gait-structure rewards and the gait gate. |
| `test_import_order.py` | Import-ordering guard for the Isaac entrypoint scripts. |
| `test_k1_actuator_gains.py` | The K1 actuator gains must be the manufacturer's, not hand-pasted over them. |
| `test_keyboard_cmd.py` | Tests for the keyboard velocity controller. |
| `test_phase_locked_stride.py` | Tests for the command-coupled gait clock and the phase-locked stride term. |
| `test_physx_gpu_capacity.py` | The GPU broadphase pair buffer must scale with env count, or contacts vanish. |
| `test_plateau_stop.py` | Tests for the reward-plateau early stopper. |
| `test_push_base_contract.py` | The frozen squat base's obs contract is 238 dims, and it must stay that way. |
| `test_push_rewards_sg.py` | Tests for the SG-style arm-contact push reward terms. |
| `test_push_scene_layout.py` | Scene-level contract for the box-push env: cell pitch and per-env ground. |
| `test_push_sg_env_cfg.py` | Wiring checks for the arm-contact SG push env cfgs. |
| `test_reward_degeneracy.py` | Static audit of the K1 reward sets for degenerate optima. |
| `test_reward_weight_ramp.py` | Tests for the upright-gated reward-weight ramp. |
| `test_squat_two_stage_curriculum.py` | Squat residual two-stage curriculum: height at zero velocity first, then velocity. |
| `test_velocity_curriculum.py` | Behaviour tests for the velocity-range curriculum. |
| `test_warm_start.py` | --warm_start must load weights without the optimizer or the iteration counter. |
