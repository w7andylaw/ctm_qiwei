# Dream to Control

**NOTE:** Check out the code for [DreamerV2](https://github.com/danijar/dreamerv2), which supports both Atari and DMControl environments.

Fast and simple implementation of the Dreamer agent in TensorFlow 2.

<img width="100%" src="https://imgur.com/x4NUHXl.gif">

If you find this code useful, please reference in your paper:

```
@article{hafner2019dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
```

## Method

![Dreamer](https://imgur.com/JrXC4rh.png)

Dreamer learns a world model that predicts ahead in a compact feature space.
From imagined feature sequences, it learns a policy and state-value function.
The value gradients are backpropagated through the multi-step predictions to
efficiently learn a long-horizon policy.

- [Project website][website]
- [Research paper][paper]
- [Official implementation][code] (TensorFlow 1)

[website]: https://danijar.com/dreamer
[paper]: https://arxiv.org/pdf/1912.01603.pdf
[code]: https://github.com/google-research/dreamer

## Instructions for this workstation

The supported GPU path is Ubuntu on WSL2. Native Windows TensorFlow can run
the code on CPU, but does not provide current NVIDIA GPU support.

Create the WSL environment once:

```bash
python3 -m venv /home/madao/.venvs/ctm-dreamer
/home/madao/.venvs/ctm-dreamer/bin/python -m pip install --upgrade pip
/home/madao/.venvs/ctm-dreamer/bin/python -m pip install -r requirements.txt
```

Run the compatibility check:

```bash
bash run_wsl.sh -u smoke_test.py
```

Train the baseline from WSL:

```bash
bash run_wsl.sh -u dreamer.py \
  --logdir ./logdir/dmc_walker_walk/dreamer/1 \
  --task dmc_walker_walk
```

Run the UAV interface smoke test from the `Dreamer-master` directory:

```bash
bash run_wsl.sh -u uav_smoke_test.py
```

The UAV tasks are available to the existing trainer as `uav_direct` and
`uav_relay`:

```bash
python dreamer.py \
  --logdir ./logdir/uav_direct/dreamer/1 \
  --task uav_direct \
  --action_repeat 1 \
  --time_limit 100
```

The adapter represents a hybrid action with `K` continuous selection channels
followed by `K` parameter channels. The selected discrete action is the argmax
of the first group, and only its matching parameter is passed to the benchmark.
It also renders the structured navigation state into the 64x64 RGB input used
by this Dreamer implementation while retaining `state`, `achieved_goal`,
`desired_goal`, and `phase` in saved episodes.

UAV runs write the following episode metrics to both TensorBoard and
`metrics.jsonl`: return, episode length, environment steps, success,
100-episode and run-wide success rates, out-of-bounds and truncation rates,
relay-reached rate, phase-switch step, MOVE/TURN/CATCH fractions, and selected
parameter magnitude. Dreamer's model, image, reward, value, and actor losses
remain available under the existing `agent/*` TensorBoard metrics.

The launcher exposes the CUDA libraries installed by
`tensorflow[and-cuda]` and keeps MuJoCo in a clean subprocess. The RTX 5070
Ti uses PTX JIT with the current TensorFlow wheel, so the first GPU operation
can take substantially longer than later runs.

Host-oriented defaults are `float32`, batch size 32, one environment process,
dataset prefetch 2, and a 100k-transition in-memory replay cache. All remain
overridable through command-line flags.

Generate plots:

```
python3 plotting.py --indir ./logdir --outdir ./plots --xaxis step --yaxis test/return --bins 3e4
```

Graphs and GIFs:

```
tensorboard --logdir ./logdir
```
