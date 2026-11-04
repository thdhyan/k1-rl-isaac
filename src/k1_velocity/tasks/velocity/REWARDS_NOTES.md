## Notes (2026-10-11)

- **New since the run above:** `feet_flat` (`gait_rewards.feet_flat_contact`, weight +1.0, std 0.15) in
  `SquatRewardsCfg`. It pays per loaded foot (contact force > 5 N) for a level sole,
  `exp(-|g_xy in foot frame|^2 / 0.15^2)`: 6° tilt keeps 61 % of the credit, 15° keeps 5 %. The old `feet_roll`
  only charged roll, so heel/toe pitch ("standing on the ankles") was free. The squat students inherit the env, so
  they get it by re-distilling from a teacher trained with it.
- The push frozen base `models/k1_push_base.pt` is a JIT export of the squat teacher (238 inputs, uses the
  privileged height scan).

---
> **Network note (2026-10-27):** the robotics router (192.168.50.1) has no internet uplink. The laptop shares its eduroam (`wlo1`) internet to the robots by NAT: `enp2s0` = 192.168.50.250/24 is the robots' gateway, with `ip_forward=1` and nft table `ip k1nat` (masquerade out `wlo1`, source 192.168.50.0/24). It is lost on laptop reboot; robots use DNS 8.8.8.8/1.1.1.1 (router DNS is wrong). Re-create steps: `docs/network_robotics_5G_2026-10-26.md` ("Robot internet via laptop").
