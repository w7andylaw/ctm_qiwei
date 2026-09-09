# Reconstruction specification (first deliverable)

## Scope and implementation plan

This directory is an isolated behavioral reproduction of Feng et al. (2023),
not a claim that the deleted implementation has been recovered.  No files in
`../Dreamer-master` are imported or modified.

Implementation order:

1. Direct planar dynamics, GoalEnv-style observations, sparse reward, seeding,
   boundary/time-limit handling, and a scripted-controller sanity check.
2. Relay pickup/delivery task with explicit phase and active-goal switching.
3. Shared PyTorch networks and distinct P-DQN/MP-DQN action evaluation.
4. Episode replay and future-strategy HER.
5. Phase-aware HER, trajectory recording, training/evaluation, and plots.
6. Multi-relay generalization after the two paper tasks reproduce reliably.

## Files

Implemented in this iteration:

- `envs/dynamics.py`: algorithm-independent UAV state transition.
- `envs/direct_navigation.py`: direct Gymnasium environment.
- `envs/relay_navigation.py`: two-stage pickup/delivery environment.
- `tests/test_dynamics.py`: numerical and control sanity checks.
- `tests/test_env.py`: direct-task, reward, boundary, seeding, and controller tests.
- `tests/test_goal_switching.py`: relay phase and completion invariants.
- `README.md`, `requirements.txt`, and `pytest.ini`: usage and reproducibility.

Implemented after the first environment iteration:

- `agents/{common,networks,pdqn,mpdqn}.py`
- `replay/replay_buffer.py`
- `replay/{her_buffer,goal_relabeling}.py` (ordinary and phase-aware future HER)
- eight experiment YAML files under `configs/`
- `scripts/{train,evaluate,run_smoke_baselines}.py`
- `tests/test_hybrid_action.py` and `tests/test_agent_env_integration.py`

Still planned, and deliberately not created as misleading empty implementations:

- `scripts/{run_all_baselines,plot_results}.py`
- `envs/wrappers.py` for normalized observations and `.npz` trajectories
- `tests/{test_hybrid_action,test_her}.py`

## State and goal spaces

The direct paper state is preserved in raw physical units:

`[x, y, speed, heading, distance_to_final, elapsed_steps]`.

The returned dictionary is:

```python
{
    "observation": state_vector,
    "achieved_goal": np.array([x, y]),
    "desired_goal": np.array([goal_x, goal_y]),
}
```

The relay paper state adds `distance_to_supply`.  This reproduction also adds
an explicit phase bit so the observation is Markov and later world models do
not have to infer a hidden pickup flag:

`[x, y, speed, heading, distance_to_final, distance_to_relay,
elapsed_steps, phase]`.

For phase 0 the active desired goal is the relay/supply point; for phase 1 it
is the final target.  The environment exposes `current_phase`, `num_phases`,
and `current_goal`.

The rich environment observation is not identical to every baseline's neural
input. P-DQN and MP-DQN receive only the six-value direct or seven-value relay
paper state. HER-PDQN and HER-MPDQN additionally receive `desired_goal` as
specified by equations (16)-(17). The explicit phase coordinate is recorded
for CT-WM but excluded from all four paper baseline encoders.

Raw observations are intentional.  Neural baselines will use a separate
normalizing wrapper so environment physics, stored trajectories, and reward
relabeling retain interpretable units.

## Hybrid action space

The Gymnasium representation is `Tuple(Discrete(K), Box(-1, 1, (1,)))`.

| Task | Index | Action | Normalized parameter | Physical meaning |
|---|---:|---|---|---|
| Both | 0 | MOVE | `[-1, 1]` | acceleration scaled by `max_acceleration` |
| Both | 1 | TURN | `[-1, 1]` | angle increment scaled by `max_turn_angle` |
| Relay | 2 | CATCH | ignored dummy slot | pickup attempt |

The dummy slot gives fixed tensor shapes while retaining the paper's fact that
CATCH has no continuous parameter.  Every parameter presented to dynamics is
clipped to `[-1, 1]`.  Defaults are 4 m/step² maximum acceleration, 40 m/step
maximum speed, and pi/3 radians maximum turn.  These physical scales were not
reported by the paper and are configurable assumptions.

## Dynamics

For action-dependent increments `delta_a` and `delta_theta`:

```text
theta[t+1] = wrap(theta[t] + delta_theta)
v[t+1]     = clip(v[t] + delta_a, 0, v_max)
x[t+1]     = x[t] + v[t+1] cos(theta[t+1]) dt
y[t+1]     = y[t] + v[t+1] sin(theta[t+1]) dt
```

MOVE sets `delta_a` and has zero turn. TURN sets `delta_theta` and has zero
acceleration. CATCH sets both to zero. Translation occurs after every action,
including CATCH, following the paper's statement that an ineffective catch
advances the UAV with current speed and angle. Altitude, steering momentum,
wind, obstacles, and energy are omitted as in the paper abstraction.

## Reward and episode boundaries

- Direct reward is 0 inside the final goal radius and -1 otherwise.
- Relay reward is 0 only after the supply has been acquired and delivered to
  the final radius; reaching/picking up the relay gives -1.
- Success terminates an episode.
- Attempted movement outside the square is clipped to the boundary and the
  episode continues. This preserves the `0/-1` reward convention and prevents
  an agent from improving return by deliberately terminating early. The legacy
  `boundary_mode="terminate"` remains available only for ablation.
- Reaching the step limit truncates, rather than terminates, the episode.
- `compute_reward` is vectorized for HER. Direct and final-delivery goals are
  positional. A phase-0 virtual pickup success additionally requires that the
  source transition execute CATCH near the relabelled goal, preventing
  MOVE/TURN from being stored as successful pickup events.

## Direct Navigation

The default map is a 2,000 m by 2,000 m square, matching the paper's 2 km by
2 km area. Start and target are uniformly sampled with a configurable minimum
separation. Initial speed is zero, initial heading is uniform, maximum length
is 100 steps, and the default success radius is 100 m.

## Relay Navigation

Three points are sampled: start, supply/relay, and final goal. Phase 0 targets
the supply. Under the paper-faithful default, the UAV must be within
`relay_radius` and execute CATCH; phase then becomes 1 without terminating.
Phase 1 targets the final goal, with the supply position abstracted as the UAV
position while carried. Only delivery can succeed. For experiments that define
"reach relay" as automatic pickup, `require_catch_action=False` switches phase
on contact; the setting must be reported with results.

## Explicit uncertainties and assumptions

| Paper specifies | Reproduction assumption | Reason |
|---|---|---|
| 2 km by 2 km square | Coordinates are metres in `[0, 2000]` | Natural physical interpretation |
| Uniform start/goal, zero initial speed | Minimum separation defaults to 400 m | Avoid reset-time successes; threshold is configurable |
| MOVE acceleration and TURN angle scaled to `[-1, 1]` | Network parameter is normalized and mapped to configurable physical limits | Paper omits unscaled limits |
| Eq. (5), immediate steering, ignored momentum | Nonnegative clipped speed, wrapped heading, `dt=1` | Paper omits speed bounds, reverse motion, and timestep |
| Ends at success/time limit; text also discusses out-of-bounds endings | Default to boundary clipping; retain termination as an ablation | A `-1` terminal boundary transition otherwise rewards deliberate early crashing relative to a `-100` timeout |
| Target/supply contact areas | Both radii default to 100 m | Exact radii are not reported |
| Relay adds CATCH with no parameter | Fixed dummy parameter slot; valid CATCH at contact changes phase | Fixed-size batched agent interface |
| Relay state has seven values but pickup state affects transitions | Expose phase as an eighth environment value, but exclude it from paper baseline encoders | Restores CT-WM observability without leaking extra information to baselines |
| GSM changes goals before/after pickup | Future HER samples are filtered to the same phase; virtual relay completion requires CATCH | Prevent invalid cross-stage goals and action-inconsistent pickup labels |
| Direct update multiplier `U=40` | Execute `episode_length * 40` updates | Matches Algorithm 1 and the direct-task parameters |
| Relay `U` varies with episode length and is limited to `[1,10]` | `U=clip(round(100/episode_length),1,10)` | The paper omits the function; inverse scaling keeps about 100 updates per episode |
| Seven runs and reported training hyperparameters | Paper configs preserve published budgets; smoke runs use explicit CLI step caps | Full relay training is 30,000 episodes per run |
| No multi-relay task is defined | Generalize the relay into N ordered CATCH stages followed by one final goal | CT-WM long-horizon extension; results must be reported separately from paper reproduction |

The paper's displayed action-space equation appears to repeat `c2` for both
actions; surrounding prose clearly assigns `c1` acceleration to MOVE and `c2`
angle to TURN, which is the interpretation used here.

## Reusable algorithm components versus new code

Can be adapted and independently verified from primary/open implementations:

- P-DQN actor/Q losses, target networks, bounded parameter gradients, replay,
  and exploration patterns.
- MP-DQN's K masked passes and diagonal Q extraction.
- HER's future-goal sampling and vectorized reward recomputation.
- Generic PyTorch initialization, Polyak updates, logging, and checkpoints.

Must be written specifically for this benchmark:

- UAV dynamics and both Gymnasium environments.
- The CATCH transition, supply-carrying abstraction, and task reward.
- GSM/phase-aware achieved-goal semantics and same-phase future filtering.
- Hybrid trajectory schema and CT-WM-facing `.npz` recorder.
- Environment sanity controllers, relay invariants, and reproduction configs.

No external implementation will be copied wholesale.  Each borrowed algorithmic
idea will be attributed and tested against the defining equations.

## CT-WM multi-relay extension

`MultiRelayNavigationEnv` supports `num_relays = 0, 1, 2, 4, 8` (and any
non-negative integer). Its observation is
`[x, y, v, theta, d_final, d_relay_1, ..., d_relay_N, step, phase]`.
The current desired goal is relay `phase` until every relay is collected, then
the final goal. Relays are ordered, so entering a later relay early has no task
effect. The sparse reward remains -1 on every non-final transition and 0 only
on successful final completion. This is a new extension, not a reconstruction
of an unreported paper environment.
