# DreamerV2 UAV migration

This build replaces the previous DreamerV1 Gaussian RSSM with the official
DreamerV2 algorithmic structure: categorical RSSM, straight-through discrete
latent samples, KL balancing, image/reward/discount heads, imagined actor-
critic learning, mixed dynamics/REINFORCE actor gradients, and a slow target
critic.

The UAV benchmark has a parameterized hybrid action space, which official
DreamerV2 does not natively support (it auto-detects purely Discrete or Box
actions). Therefore one explicit extension is necessary: `HybridActionDecoder`
uses a categorical action selection plus continuous per-action parameters.
This is not claimed to be part of upstream DreamerV2. The external
`uav_adapter.py` file is removed; the environment contract is implemented in
`envs.py` as `DreamerV2UAVEnv`.

Default task: `uav_relay`; time limit 100; action repeat 1; batch length 20.
The batch length differs from upstream default 50 because paper UAV episodes
can terminate before 50 steps.

Run:

    python dreamer.py --task uav_relay --logdir ./outputs/dv2_relay

Task 1:

    python dreamer.py --task uav_direct --logdir ./outputs/dv2_direct
