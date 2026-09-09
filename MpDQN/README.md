# HER-MPDQN UAV benchmark reproduction

An isolated, reproducible reconstruction of **“Reinforcement learning with
parameterized action space and sparse reward for UAV navigation”** (Feng et
al., 2023). The original repository is unavailable, so this project separates
reported facts from configurable reconstruction assumptions.

The Direct/Relay environments, all four baselines, replay/HER, fixed-step
experiment tooling, deterministic evaluation, plotting, and the configurable
multi-relay extension are implemented and tested. Results are behavioral
reproduction evidence, not a claim of exact numerical recovery of deleted code.

## Implemented environments

- **Direct Navigation:** uniformly sampled start/target in a continuous square,
  paper-style MOVE/TURN parameterized actions, sparse `0/-1` reward, boundary
  clipping, and time-limit truncation.
- **Relay Navigation:** start -> supply pickup -> final delivery, an explicit
  phase and active goal, paper-style CATCH action, and no intermediate reward.
- **Multi-Relay Navigation:** ordered 0/1/2/4/8 relay stages, exposed
  `current_phase`, `num_phases`, and `current_goal`, with unchanged sparse reward.
- **Goal API:** dictionary observations with `observation`, `achieved_goal`, and
  `desired_goal`, plus vectorized `compute_reward` for future HER.
- **P-DQN:** joint action-parameter Q input, deterministic parameter actor,
  target networks, Q/actor losses, exploration, Polyak updates, and checkpoints.
- **MP-DQN:** K masked parameter passes with diagonal Q extraction; it is not an
  alias for P-DQN and has a dedicated irrelevant-parameter invariance test.
- **Replay:** fixed-capacity circular storage, deterministic sampling,
  terminated/truncated separation, and compressed save/restore.
- **Ordinary HER:** configurable unique future-goal sampling, desired-goal
  replacement, vectorized environment reward reuse, and relabel counters.
- **Phase-aware HER:** paper-GSM reconstruction that admits future goals only
  from the source phase, requires CATCH-consistent virtual pickup success, and
  reports every rejected cross-phase candidate.
- **Paper-facing inputs:** P-DQN/MP-DQN consume only the published state;
  HER variants consume `(state, goal)`. The environment still exposes phase to
  trajectory recorders, but phase is not leaked into baseline networks.

The complete state/action equations, assumptions, source-to-code decisions,
file plan, and Stage 3-7 boundaries are in [RECONSTRUCTION_SPEC.md](RECONSTRUCTION_SPEC.md).

For the previous full-run analysis, current v2 pilot results, and confirmed
environment/HER issues, see [EXPERIMENT_ANALYSIS_AND_SCENARIO_ISSUES.md](../EXPERIMENT_ANALYSIS_AND_SCENARIO_ISSUES.md)
(Chinese, 2026-09-06). These issues remain open; successful execution alone
does not establish benchmark correctness.

## Install and test

From this directory:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

Minimal use:

```python
import numpy as np
from envs import DirectNavigationEnv

env = DirectNavigationEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(
    (0, np.array([0.5], dtype=np.float32))  # MOVE, half acceleration
)
```

## Baseline training

```bash
python scripts/train.py --config configs/direct_her_mpdqn.yaml
python scripts/run_smoke_baselines.py --environment both --episodes 3
python scripts/run_direct_pilot.py --steps 5000 --eval-episodes 30
python scripts/run_all_baselines.py --environments direct,relay \
  --seeds 0,1,2 --steps 5000 --eval-episodes 30 --workers 8
# Paper episode budgets and 1000-episode evaluation (no --steps cap):
python scripts/run_all_baselines.py --environments direct,relay \
  --seeds 0,1,2 --eval-episodes 1000 --workers 8 --worker-threads 1
python scripts/plot_results.py \
  --input-root outputs/reproduction_3seed_5000
python scripts/run_horizon_experiments.py \
  --relay-counts 0,1,2,4,8 --seeds 0,1,2 --steps 5000
python scripts/evaluate.py \
  --checkpoint outputs/direct_her_mpdqn/seed_0/best.pt \
  --episodes 100 --seed 10000 --save-trajectories
```

Each episode appends return, success, length, environment steps, replay size,
Q loss, parameter-actor loss, HER relabel count, boundary-hit rate, and cross-phase filter count
to `metrics.jsonl`. Resolved YAML plus `best.pt`, deterministic-validation
`best_eval.pt`, and `last.pt` are written to
the configured output directory. Evaluation writes deterministic per-episode
and aggregate metrics and can export complete CT-WM-ready `.npz` trajectories.
`run_all_baselines.py` writes per-run records plus JSON/CSV mean and population
standard deviation across seeds. `plot_results.py` produces seed-aggregated
success curves and deterministic-evaluation success bars. Use a larger
`--steps` value for definitive studies; 5,000 steps is a bounded pilot budget,
whereas the paper reports much longer episode-based training.

## Checkpoints and resuming

`best.pt`, `last.pt`, and every `checkpoint_<episode>.pt` are full training
checkpoints. They contain online/target networks, both optimizers, replay/HER
storage and counters, agent/replay/HER/environment RNG states, global Python,
NumPy and PyTorch RNG states, update credit, step/episode counters, rolling
success history, and the last losses. Writes are atomic through a temporary
file. Checkpoints are saved at completed-episode boundaries.

Resume a single run while increasing its total budget:

```bash
python scripts/train.py --config configs/relay_her_mpdqn.yaml \
  --steps 3000000 --resume outputs/relay_her_mpdqn/seed_0/checkpoint_500.pt
```

The requested `episodes`/`steps` are total targets, not additional amounts.
Environment, agent, replay, algorithm, and seed settings must match; training
budget and output directory may change. Resuming in the same directory trims
stale metric rows after the selected checkpoint before appending. For a suite,
rerun `run_all_baselines.py` with `--reuse-existing`; completed evaluations are
skipped and incomplete runs resume from their highest periodic checkpoint.
Because replay storage is included, full checkpoints are substantially larger
than inference-only model files; adjust `checkpoint_every` according to disk
budget for long multi-seed runs.

`--workers N` runs N complete experiments in independent spawned processes.
Networks, replay buffers, optimizers, RNG streams, checkpoints, and output
directories remain isolated, so concurrency changes scheduling but not the
per-run update-to-data ratio. These small MLPs benchmark faster on this
workstation's CPU than its GPU; the paper configs therefore use CPU and the
runner defaults to at most eight one-thread workers. `--worker-threads` can be
tuned for other hardware.

## Paper update schedule

The direct configs implement the published `episode_length * U` optimizer
updates with `U=40`. The paper says relay `U` varies with episode length and is
clipped to `[1, 10]`, but does not publish the function. This reproduction uses
the explicit assumption `U=clip(round(100 / episode_length), 1, 10)`, which
keeps approximately 100 optimizer updates per complete relay episode. Legacy
configs without `update_schedule` retain interval/gradient-step behavior for
tests and ablations. Gradient accumulation is not substituted for paper
updates because it changes Adam and target-network update semantics.

## Long-horizon extension

The extension is intentionally separate from the original benchmark:

```python
from envs import MultiRelayNavigationEnv
env = MultiRelayNavigationEnv(num_relays=8)
```

The agent-facing observation grows with the relay count and includes distances
to the final goal and every relay, along with step and phase. Relays must be
visited in order and require CATCH when configured. Only final-goal completion
returns `0`; relay completion remains `-1`, so reward density is not increased.
The provided `configs/multi_relay_her_mpdqn.yaml` is a starting point rather
than an original-paper configuration.

## Algorithm boundaries

P-DQN uses one joint parameter-vector pass. MP-DQN instead constructs
one masked vector per discrete action and uses only the diagonal Q outputs, so
irrelevant parameters cannot affect the selected action's input. HER uses
future achieved goals. Relay HER will sample future goals only within the same
phase, implementing the paper's goal-switch mechanism rather than relabeling
blindly across pickup and delivery.

## Sources

- Feng, S., Li, X., Ren, L., & Xu, S. (2023). [Reinforcement learning with
  parameterized action space and sparse reward for UAV navigation](https://doi.org/10.20517/ir.2023.10).
- Xiong et al. (2018). [Parametrized deep Q-networks learning: Reinforcement
  learning with discrete-continuous hybrid action space](https://doi.org/10.48550/arXiv.1810.06394).
- Bester, James, & Konidaris (2019). [Multi-Pass Q-Networks for Deep
  Reinforcement Learning with Parameterised Action Spaces](https://arxiv.org/abs/1905.04388).
- Andrychowicz et al. (2017). [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495).

The benchmark paper explicitly reports the 2 km square, state vectors,
MOVE/TURN/CATCH definitions, Eq. (5) dynamics, `0/-1` task rewards, 100-step
direct limit, and goal switching. It does not report several physical scales
or contact radii; all such choices are listed in the reconstruction spec.
