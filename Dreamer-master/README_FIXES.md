# Navigation world-model corrections

This version changes the Dreamer observation/reward contract. Start a new run;
do not reuse old image-only replay or checkpoints. The default output directory
is `outputs/dreamerv2_uav_relay_vector_v2`. Original base navigation environments
keep their old rewards for separate paper comparisons; DreamerV2UAVEnv uses
success=1 and otherwise=0. All compared agents must use the same observation,
reward and termination contract for a controlled comparison.

## Changes

- Imagination returns H+1 states and H actions. Actor log probabilities and
  critic predictions use source states; rewards and continuation use arrival
  states. Lambda returns bootstrap from the final state. True terminal replay
  starts have zero actor/critic weight.
- Actions remain 2*K wide, but only the chosen MOVE/TURN parameter is nonzero.
  CATCH has no parameter. This applies to sampling, evaluation, random prefill,
  replay preprocessing and the RSSM input. Ignored parameter gradients are zero.
- The world model uses a 13-coordinate normalized vector with an MLP encoder
  and vector reconstruction head. Images remain available for diagnostics but
  are not training inputs or reconstruction targets.
- Vector order: x, y, speed, sin(heading), cos(heading), final_dx, final_dy,
  supply_dx, supply_dy, elapsed, phase, active_dx, active_dy. Coordinates x/y,
  speed and elapsed are scaled to [-1,1]; offsets are divided by map size.
  Supply offsets refer to the pickup location. Direct tasks use zero supply
  offsets and phase=0. Out-of-bounds observations are not clipped.
- Dreamer reward is one only on successful, in-bounds completion, zero on
  failure, timeout and intermediate steps. Early failure and timeout both return
  zero; successful completion returns a positive discounted return. No distance
  shaping or intermediate pickup bonus is added. Timeout bootstrap semantics
  remain unchanged.

## Regression test

Run `python test_navigation_fixes.py` with TensorFlow, TensorFlow Probability,
Gymnasium, NumPy, Pillow and tqdm installed. Checks include source/action
alignment, lambda-return indexing, unused-parameter gradients and RSSM
invariance, vector observability, failure/timeout/success rewards, and small
vector-only training plus inference runs for both task types.

Example new run: `python dreamer.py --task uav_relay --parallel none`

These corrections do not establish a convergence or success-rate improvement;
that requires fresh multi-seed training under the revised contract.

## Replay sampling fix

- Episodes shorter than `batch_length` are excluded before random sampling.
  Files are preserved; valid episodes are still sampled uniformly.
- A `Replay filter` summary is printed only when the excluded set changes,
  not every time the same short episode would have been drawn.
- An empty replay or replay with no eligible episodes raises a clear error
  instead of looping indefinitely. Capacity still limits the newest replay
  window before eligibility filtering, preserving existing capacity semantics.
- Training rescans every `max(batch_size, train_steps)` yielded sequences
  (50 with the default configuration), not every single sequence.
- Run `python -m unittest test_replay_sampling test_navigation_fixes -v`.
  Restart the Python training process to load changed code. Existing replay
  files remain compatible with this sampler change. The current entry point
  does not restore saved network weights automatically, so restarting is not
  an exact checkpoint resume.
