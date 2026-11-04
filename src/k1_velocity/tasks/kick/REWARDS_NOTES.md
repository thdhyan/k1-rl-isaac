## Notes (2026-10-11)

- The goal is fixed at `GOAL_POS = (FIELD_L/2, 0, GOAL_H/2)` (`kick/mdp.py`); `curriculum = None`. Ball resets
  come from `reset_ball_omnireset`: at_ball_shoot 50 %, stand_ready 30 %, walk_up 20 %.
- Final run: 3399 iterations, about 53 % goals.
- Proposed: goal-position and ball-distance curriculum (random goal y and distance, widening on success rate).

---
> **Network note (2026-10-27):** the robotics router (192.168.50.1) has no internet uplink. The laptop shares its eduroam (`wlo1`) internet to the robots by NAT: `enp2s0` = 192.168.50.250/24 is the robots' gateway, with `ip_forward=1` and nft table `ip k1nat` (masquerade out `wlo1`, source 192.168.50.0/24). It is lost on laptop reboot; robots use DNS 8.8.8.8/1.1.1.1 (router DNS is wrong). Re-create steps: `docs/network_robotics_5G_2026-10-26.md` ("Robot internet via laptop").
