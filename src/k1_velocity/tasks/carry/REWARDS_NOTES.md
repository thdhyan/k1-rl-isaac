## Notes (2026-10-11)

- The goal radius IS on a curriculum: `goal_radius_curriculum` grows from 1 to 4 m in 0.25 m steps while falls stay
  under 8 % of resets. Run `p7_carry/2026-10-10_17-27-51` hit the 4 m cap at about it1500, with success rate 1.0
  and falls about 1 % after that, so the cap (not the policy) is now the limit.
- Play bug, fixed 2026-10-11: `K1CarryEnvCfg_PLAY` re-ran the schedule from 1 m, so videos showed near goals. It
  now starts at the 4 m target.
- Next: raise `target` (for example 8 m; needs `env_spacing` > 2 × target, so 18 m) and add a `goal_reached`
  term that scales with distance beyond 4 m.
- Architecture: decoupled actor (legs head + arms head, `decoupled_policy.py`). Not residual.

---
> **Network note (2026-10-27):** the robotics router (192.168.50.1) has no internet uplink. The laptop shares its eduroam (`wlo1`) internet to the robots by NAT: `enp2s0` = 192.168.50.250/24 is the robots' gateway, with `ip_forward=1` and nft table `ip k1nat` (masquerade out `wlo1`, source 192.168.50.0/24). It is lost on laptop reboot; robots use DNS 8.8.8.8/1.1.1.1 (router DNS is wrong). Re-create steps: `docs/network_robotics_5G_2026-10-26.md` ("Robot internet via laptop").
