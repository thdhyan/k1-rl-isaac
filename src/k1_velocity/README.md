# src/k1_velocity

Part of [k1-rl-isaac](https://github.com/thdhyan/k1-rl-isaac).

| File | What it does |
|---|---|
| `__init__.py` | K1 velocity locomotion task for Isaac Lab. |
| `assets_mjlab.py` | K1 as trained in mjlab (``mjlab_k1``), for Isaac Lab: soft Booster gains from the exported USD. |
| `register_tasks.py` | Gym-registration callback for the unified Isaac Lab 3.0 CLI. |
| `shove_mdp.py` | Force-variant (P1f / P2f) custom MDP terms. |
| `sim_backend.py` | Physics backend selection for K1 tasks. |
