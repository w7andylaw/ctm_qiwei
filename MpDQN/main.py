"""Unified experiment entry point for HER-MPDQN reproduction.

Defaults: run all baselines ON; smoke, direct pilot, and horizon experiments OFF.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- Training ---------------------------------------------------------------
from agents import (ActionSelection, MPDQNAgent, PDQNAgent, PDQNConfig, save_checkpoint)
from envs import (DirectNavigationEnv, MultiRelayNavigationEnv, RelayNavigationEnv, GoalObservationEncoder)
from replay import (  # noqa: E402
    GoalTransition,
    HERReplayBuffer,
    PhaseAwareHERReplayBuffer,
    ReplayBuffer,
    ReplayBufferConfig,
)


ALGORITHMS = ("pdqn", "mpdqn", "her_pdqn", "her_mpdqn")
ENVIRONMENTS = ("direct", "relay", "multi_relay")
TRAINING_CHECKPOINT_VERSION = 2


# -----------------------------------------------------------------------------
# Built-in experiment configuration (replaces configs/*.yaml)
# -----------------------------------------------------------------------------
COMMON_AGENT = {
    "gamma": 0.99,
    "tau": 0.01,
    "gradient_clip_norm": 10.0,
    "device": "cpu",
}
COMMON_EXPLORATION = {
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "parameter_noise_start": 0.5,
    "parameter_noise_end": 0.05,
}

TASK_CONFIGS = {
    "direct": {
        "env": {
            "boundary_mode": "terminate",
            "map_size": 2000.0,
            "goal_radius": 100.0,
            "max_episode_steps": 100,
            "min_start_goal_distance": 0.0,
        },
        "agent": {
            **COMMON_AGENT,
            "hidden_sizes": [128, 64],
            "q_learning_rate": 0.01,
            "parameter_learning_rate": 0.001,
        },
        "replay": {"capacity": 50_000},
        "training": {
            "episodes": 2_000,
            "learning_starts": 1_000,
            "batch_size": 128,
            "update_schedule": {"mode": "paper_fixed", "multiplier": 40},
            "checkpoint_every": 200,
            "validation_every": 200,
            "validation_episodes": 100,
        },
        "exploration": {**COMMON_EXPLORATION, "decay_steps": 100_000},
    },
    "relay": {
        "env": {
            "boundary_mode": "terminate",
            "map_size": 2000.0,
            "goal_radius": 100.0,
            "relay_radius": 100.0,
            "max_episode_steps": 100,
            "min_start_goal_distance": 0.0,
            "require_catch_action": True,
        },
        "agent": {
            **COMMON_AGENT,
            "hidden_sizes": [256, 128, 64],
            "q_learning_rate": 0.001,
            "parameter_learning_rate": 0.00001,
        },
        "replay": {"capacity": 150_000},
        "training": {
            "episodes": 30_000,
            "learning_starts": 2_000,
            "batch_size": 128,
            # Paper only reports U varies with episode length and is clipped to [1, 10].
            # This explicit mapping remains a reconstruction assumption.
            "update_schedule": {
                "mode": "paper_dynamic",
                "reference_episode_length": 100,
                "min_multiplier": 1,
                "max_multiplier": 10,
            },
            "checkpoint_every": 2_000,
            "validation_every": 2_000,
            "validation_episodes": 100,
        },
        "exploration": {**COMMON_EXPLORATION, "decay_steps": 500_000},
    },
    "multi_relay": {
        "env": {
            "boundary_mode": "clip",
            "num_relays": 1,
            "map_size": 2000.0,
            "goal_radius": 100.0,
            "relay_radius": 100.0,
            "max_episode_steps": 200,
            "min_start_goal_distance": 200.0,
            "require_catch_action": True,
        },
        "agent": {
            **COMMON_AGENT,
            "hidden_sizes": [256, 128, 64],
            "q_learning_rate": 0.001,
            "parameter_learning_rate": 0.00001,
        },
        "replay": {"capacity": 300_000},
        "training": {
            "episodes": 30_000,
            "learning_starts": 2_000,
            "batch_size": 128,
            "update_schedule": {
                "mode": "paper_dynamic",
                "reference_episode_length": 100,
                "min_multiplier": 1,
                "max_multiplier": 10,
            },
            "checkpoint_every": 2_000,
            "validation_every": 2_000,
            "validation_episodes": 100,
        },
        "exploration": {**COMMON_EXPLORATION, "decay_steps": 500_000},
    },
}


def get_config(environment: str, algorithm: str, *, seed: int = 0,
               output_dir: str | None = None) -> dict[str, Any]:
    """Construct one experiment config entirely from inherited Python defaults."""
    if environment not in TASK_CONFIGS:
        raise ValueError(f"Unknown environment {environment!r}")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}")
    cfg = copy.deepcopy(TASK_CONFIGS[environment])
    cfg["experiment"] = {
        "algorithm": algorithm,
        "environment": environment,
        "seed": int(seed),
        "output_dir": output_dir or f"outputs/{environment}_{algorithm}/seed_{seed}",
    }
    uses_her = algorithm.startswith("her_")
    cfg["replay"]["her_k"] = 4 if uses_her else 0
    # Strict paper tasks use GSM goal semantics in the environment, not the
    # extra same-phase HER filter. Keep phase-aware filtering only for extension work.
    cfg["replay"]["phase_aware"] = bool(environment == "multi_relay" and uses_her)
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a resolved JSON config saved beside a checkpoint."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Saved config must be a JSON object")
    config["_config_path"] = str(path.resolve())
    return config


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    for section in ("experiment", "env", "agent", "replay", "training"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing config section: {section}")
    algorithm = str(config["experiment"].get("algorithm"))
    environment = str(config["experiment"].get("environment"))
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}")
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment {environment!r}")
    phase_aware = bool(config["replay"].get("phase_aware", False))
    uses_her = algorithm.startswith("her_")
    if phase_aware and not uses_her:
        raise ValueError("phase_aware replay requires a HER algorithm")
    if phase_aware and environment not in ("relay", "multi_relay"):
        raise ValueError("phase_aware replay is only valid for staged relay tasks")
    if int(config["training"].get("episodes", 0)) <= 0:
        raise ValueError("training.episodes must be positive")
    max_steps = config["training"].get("max_environment_steps")
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("training.max_environment_steps must be positive when set")
    schedule = config["training"].get("update_schedule")
    if schedule is not None:
        if not isinstance(schedule, Mapping):
            raise ValueError("training.update_schedule must be a mapping")
        mode = str(schedule.get("mode"))
        if mode not in ("paper_fixed", "paper_dynamic"):
            raise ValueError("Unknown training.update_schedule mode")
    for name in ("validation_every", "validation_episodes"):
        value = int(config["training"].get(name, 0))
        if value < 0:
            raise ValueError(f"training.{name} cannot be negative")


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return name


def make_environment(config: Mapping[str, Any]):
    name = config["experiment"]["environment"]
    values = dict(config["env"])
    if name == "direct":
        return DirectNavigationEnv(**values)
    if name == "relay":
        return RelayNavigationEnv(**values)
    return MultiRelayNavigationEnv(**values)


def parameter_sizes_for_environment(env) -> tuple[int, ...]:
    """Map the environment's discrete actions to their scalar parameter sizes."""
    if env.action_space[0].n == 2:
        return (1, 1)
    if env.action_space[0].n == 3:
        return (1, 1, 0)
    raise ValueError(f"Unsupported number of hybrid actions: {env.action_space[0].n}")


def make_observation_encoder(
    config: Mapping[str, Any], env,
) -> GoalObservationEncoder:
    """Build the paper-facing input while preserving richer environment data.

    Only HER variants receive the explicit goal. Relay phase remains available
    through the environment/recorder API but is excluded from all paper
    baseline network inputs because it is not part of the published state.
    """
    algorithm = str(config["experiment"]["algorithm"])
    environment = str(config["experiment"]["environment"])
    excluded = (-1,) if environment in ("relay", "multi_relay") else ()
    return GoalObservationEncoder(
        env.observation_space,
        include_goal=algorithm.startswith("her_"),
        excluded_state_indices=excluded,
    )


def make_agent(config: Mapping[str, Any], state_dim: int, parameter_sizes: tuple[int, ...]):
    values = config["agent"]
    agent_config = PDQNConfig(
        state_dim=state_dim,
        parameter_sizes=parameter_sizes,
        hidden_sizes=tuple(int(size) for size in values["hidden_sizes"]),
        gamma=float(values["gamma"]),
        tau=float(values["tau"]),
        q_learning_rate=float(values["q_learning_rate"]),
        parameter_learning_rate=float(values["parameter_learning_rate"]),
        gradient_clip_norm=float(values.get("gradient_clip_norm", 10.0)),
        seed=int(config["experiment"]["seed"]),
        device=resolve_device(str(values.get("device", "auto"))),
    )
    algorithm = config["experiment"]["algorithm"]
    return (MPDQNAgent if "mpdqn" in algorithm else PDQNAgent)(agent_config)


def make_replay(
    config: Mapping[str, Any],
    state_dim: int,
    parameter_dim: int,
    num_actions: int,
    encoder: GoalObservationEncoder,
):
    values = config["replay"]
    replay_config = ReplayBufferConfig(
        capacity=int(values["capacity"]),
        state_dim=state_dim,
        parameter_dim=parameter_dim,
        num_actions=num_actions,
        seed=int(config["experiment"]["seed"]),
    )
    algorithm = config["experiment"]["algorithm"]
    if not algorithm.startswith("her_"):
        return ReplayBuffer(replay_config)
    # The paper's GSM changes achieved/desired-goal semantics according to
    # pickup state. It does not specify same-phase future filtering. Therefore
    # strict direct/relay reproduction uses ordinary future-strategy HER.
    # Phase-aware filtering is retained only for the non-paper multi-relay extension.
    use_phase_filter = (
        config["experiment"]["environment"] == "multi_relay"
        and bool(values.get("phase_aware", False))
    )
    buffer_class = PhaseAwareHERReplayBuffer if use_phase_filter else HERReplayBuffer
    return buffer_class(
        replay_config,
        her_k=int(values.get("her_k", 4)),
        state_encoder=encoder,
    )


def apply_paper_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only settings explicitly reported by Feng et al. (2023).

    Unreported quantities (physical control scales, target/contact radii,
    gamma, tau, exploration schedule, and the exact relay U(length) mapping)
    are deliberately left configurable rather than silently invented.
    """
    cfg = copy.deepcopy(dict(config))
    env_name = str(cfg["experiment"]["environment"])
    algorithm = str(cfg["experiment"]["algorithm"])
    if env_name in ("direct", "relay"):
        cfg["env"]["boundary_mode"] = "terminate"
        cfg["env"]["min_start_goal_distance"] = 0.0
        # Paper samples positions uniformly over the full 2 km x 2 km square.
        cfg["env"]["map_size"] = 2000.0
    if env_name == "direct":
        cfg["env"]["max_episode_steps"] = 100
        cfg["agent"]["hidden_sizes"] = [128, 64]
        cfg["agent"]["q_learning_rate"] = 1e-2
        cfg["agent"]["parameter_learning_rate"] = 1e-3
        cfg["replay"]["capacity"] = 50_000
        cfg["training"]["batch_size"] = 128
        cfg["training"]["episodes"] = 2_000
        cfg["training"]["update_schedule"] = {"mode": "paper_fixed", "multiplier": 40}
    elif env_name == "relay":
        cfg["agent"]["hidden_sizes"] = [256, 128, 64]
        cfg["agent"]["q_learning_rate"] = 1e-3
        cfg["agent"]["parameter_learning_rate"] = 1e-5
        cfg["replay"]["capacity"] = 150_000
        cfg["training"]["episodes"] = 30_000
        # The paper states only that U varies with episode length and lies in
        # [1, 10]; it does not publish the mapping. Keep the configured mapping.
    if algorithm.startswith("her_"):
        cfg["replay"]["her_k"] = 4
    return cfg


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    if duration <= 0:
        return float(end)
    fraction = min(max(step / duration, 0.0), 1.0)
    return float(start + fraction * (end - start))


def paper_update_count(
    training: Mapping[str, Any], *, episode_length: int, eligible_steps: int,
) -> tuple[int, int]:
    """Return optimizer updates and the paper's per-step multiplier U.

    Configurations without ``update_schedule`` retain the original interval
    behavior for small tests and third-party configs.
    """
    schedule = training.get("update_schedule")
    if not schedule:
        update_every = int(training.get("update_every", 1))
        if update_every <= 0:
            raise ValueError("training.update_every must be positive")
        events = eligible_steps // update_every
        return events * int(training.get("gradient_steps", 1)), 0
    mode = str(schedule.get("mode"))
    if mode == "paper_fixed":
        multiplier = int(schedule.get("multiplier", 1))
    elif mode == "paper_dynamic":
        reference = float(schedule.get("reference_episode_length", 100))
        minimum = int(schedule.get("min_multiplier", 1))
        maximum = int(schedule.get("max_multiplier", 10))
        if reference <= 0 or minimum <= 0 or maximum < minimum:
            raise ValueError("Invalid paper_dynamic update schedule")
        multiplier = int(np.clip(round(reference / max(1, episode_length)), minimum, maximum))
    else:
        raise ValueError(f"Unknown training.update_schedule mode: {mode!r}")
    if multiplier <= 0:
        raise ValueError("Paper update multiplier must be positive")
    return eligible_steps * multiplier, multiplier


def random_selection(agent) -> ActionSelection:
    parameters = agent.rng.uniform(
        -1.0, 1.0, size=agent.spec.total_parameter_dim).astype(np.float32)
    action = int(agent.rng.integers(agent.spec.num_actions))
    return ActionSelection(
        discrete_action=action,
        selected_parameter=agent.spec.selected_parameter(parameters, action),
        all_parameters=parameters,
        q_values=np.zeros(agent.spec.num_actions, dtype=np.float32),
    )


def deterministic_validation(
    config: Mapping[str, Any], agent, encoder: GoalObservationEncoder,
    *, episodes: int, seed: int,
) -> dict[str, float]:
    """Evaluate without perturbing the training environment or exploration RNG."""
    validation_env = make_environment(config)
    rng_state = copy.deepcopy(agent.rng.bit_generator.state)
    successes: list[float] = []
    returns: list[float] = []
    relay_reached: list[float] = []
    try:
        obs, _ = validation_env.reset(seed=seed)
        for episode in range(episodes):
            if episode:
                obs, _ = validation_env.reset()
            episode_return = 0.0
            reached = False
            while True:
                source_phase = validation_env.current_phase
                selection = agent.select_action(
                    encoder(obs), epsilon=0.0, parameter_noise_std=0.0)
                obs, reward, terminated, truncated, info = validation_env.step(
                    selection.environment_action())
                episode_return += reward
                reached = reached or validation_env.current_phase > source_phase
                if terminated or truncated:
                    break
            successes.append(float(bool(info.get("is_success", False))))
            returns.append(float(episode_return))
            relay_reached.append(float(reached or validation_env.current_phase > 0))
    finally:
        agent.rng.bit_generator.state = rng_state
        validation_env.close()
    return {
        "validation_success_rate": float(np.mean(successes)),
        "validation_mean_return": float(np.mean(returns)),
        "validation_relay_reached_rate": float(np.mean(relay_reached)),
    }


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def resume_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Configuration fields that must not change across a resumed run."""
    return {
        "experiment": {
            "algorithm": config["experiment"]["algorithm"],
            "environment": config["experiment"]["environment"],
            "seed": int(config["experiment"]["seed"]),
        },
        "env": copy.deepcopy(dict(config["env"])),
        "agent": copy.deepcopy(dict(config["agent"])),
        "replay": copy.deepcopy(dict(config["replay"])),
    }


def save_training_checkpoint(
    path: Path, *, agent, replay, env, config: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
) -> None:
    payload = agent.checkpoint()
    payload.update({
        "training_checkpoint_version": TRAINING_CHECKPOINT_VERSION,
        "resume_signature": resume_signature(config),
        "replay_state": replay.state_dict(),
        "trainer_state": dict(trainer_state),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "environment": copy.deepcopy(env.np_random.bit_generator.state),
        },
    })
    save_checkpoint(path, payload)


def load_training_checkpoint(
    path: Path, *, agent, replay, env, config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    if int(payload.get("training_checkpoint_version", -1)) != TRAINING_CHECKPOINT_VERSION:
        raise ValueError(
            "Checkpoint contains agent weights only and cannot resume training; "
            "use a checkpoint produced by the updated trainer")
    if payload.get("resume_signature") != resume_signature(config):
        raise ValueError(
            "Resume checkpoint is incompatible with the current environment, agent, replay, or seed config")
    agent.load(path)
    replay.load_state_dict(payload["replay_state"])
    rng = payload["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
    env.np_random.bit_generator.state = copy.deepcopy(rng["environment"])
    state = dict(payload["trainer_state"])
    if int(state["total_updates"]) != agent.update_steps:
        raise ValueError("Trainer and agent update counters disagree in checkpoint")
    return state


def truncate_metrics_to_checkpoint(
    path: Path, *, completed_episodes: int, total_steps: int,
) -> None:
    """Remove stale records after the selected checkpoint before appending."""
    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if (int(record["episode"]) <= completed_episodes
                and int(record["environment_steps"]) <= total_steps):
            retained.append(json.dumps(record, sort_keys=True) + "\n")
    path.write_text("".join(retained), encoding="utf-8")


def train(
    config: Mapping[str, Any], *, overrides: Mapping[str, Any] | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    config = deep_update(dict(config), overrides or {})
    config = apply_paper_protocol(config)
    validate_config(config)
    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_environment(config)
    obs, _ = env.reset(seed=seed)
    encoder = make_observation_encoder(config, env)
    parameter_sizes = parameter_sizes_for_environment(env)
    agent = make_agent(config, encoder.output_dim, parameter_sizes)
    replay = make_replay(
        config,
        encoder.output_dim,
        agent.spec.total_parameter_dim,
        agent.spec.num_actions,
        encoder,
    )

    output_dir = Path(config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    resolved_config = copy.deepcopy(config)
    resolved_config.pop("_config_path", None)

    training = config["training"]
    exploration = config.get("exploration", {})
    total_steps = 0
    total_updates = 0
    total_successes = 0
    completed_episodes = 0
    update_credit = 0
    success_window_size = 10 if config["experiment"]["environment"] == "direct" else 100
    success_window: deque[float] = deque(maxlen=success_window_size)
    last_losses: dict[str, float] = {}
    best_success_rate = -1.0
    best_validation_success = -1.0
    best_validation_return = -float("inf")

    if resume_from is None:
        metrics_path.write_text("", encoding="utf-8")
        start_episode = 1
        reset_before_first_episode = False
    else:
        resume_path = Path(resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        restored = load_training_checkpoint(
            resume_path, agent=agent, replay=replay, env=env, config=config)
        total_steps = int(restored["total_steps"])
        total_updates = int(restored["total_updates"])
        total_successes = int(restored["total_successes"])
        completed_episodes = int(restored["completed_episodes"])
        update_credit = int(restored["update_credit"])
        success_window.extend(float(value) for value in restored["success_window"])
        last_losses = dict(restored["last_losses"])
        best_success_rate = float(restored["best_success_rate"])
        best_validation_success = float(restored.get("best_validation_success", -1.0))
        best_validation_return = float(restored.get("best_validation_return", -float("inf")))
        start_episode = completed_episodes + 1
        reset_before_first_episode = True
        source_metrics = resume_path.parent / "metrics.jsonl"
        if (not metrics_path.exists() and source_metrics.exists()
                and source_metrics.resolve() != metrics_path.resolve()):
            metrics_path.write_text(
                source_metrics.read_text(encoding="utf-8"), encoding="utf-8")
        truncate_metrics_to_checkpoint(
            metrics_path, completed_episodes=completed_episodes, total_steps=total_steps)

    # Write only after a requested resume has passed compatibility checks, so a
    # bad resume command cannot overwrite the original run configuration.
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True), encoding="utf-8")

    def checkpoint_state() -> dict[str, Any]:
        return {
            "total_steps": total_steps,
            "total_updates": total_updates,
            "total_successes": total_successes,
            "completed_episodes": completed_episodes,
            "update_credit": update_credit,
            "success_window": list(success_window),
            "last_losses": dict(last_losses),
            "best_success_rate": best_success_rate,
            "best_validation_success": best_validation_success,
            "best_validation_return": best_validation_return,
        }

    def save_full_checkpoint(path: Path) -> None:
        save_training_checkpoint(
            path, agent=agent, replay=replay, env=env, config=config,
            trainer_state=checkpoint_state())

    episode_bar = tqdm(
        range(start_episode, int(training["episodes"]) + 1),
        desc=f"Train {config['experiment']['environment']}/{config['experiment']['algorithm']} seed={seed}",
        unit="ep",
        dynamic_ncols=True,
    )
    for episode_index in episode_bar:
        max_environment_steps = training.get("max_environment_steps")
        if max_environment_steps is not None and total_steps >= int(max_environment_steps):
            break
        if reset_before_first_episode or episode_index > start_episode:
            obs, _ = env.reset()
        reset_before_first_episode = False
        steps_before_episode = total_steps
        episode_transitions: list[GoalTransition] = []
        episode_return = 0.0
        episode_losses: list[dict[str, float]] = []
        final_info: dict[str, Any] = {}

        while True:
            state = encoder(obs)
            epsilon = linear_schedule(
                float(exploration.get("epsilon_start", 1.0)),
                float(exploration.get("epsilon_end", 0.05)),
                int(exploration.get("decay_steps", 100_000)),
                total_steps,
            )
            noise = linear_schedule(
                float(exploration.get("parameter_noise_start", 0.5)),
                float(exploration.get("parameter_noise_end", 0.05)),
                int(exploration.get("decay_steps", 100_000)),
                total_steps,
            )
            if total_steps < int(training["learning_starts"]):
                selection = random_selection(agent)
            else:
                selection = agent.select_action(
                    state, epsilon=epsilon, parameter_noise_std=noise)
            source_phase = env.current_phase
            next_obs, reward, terminated, truncated, info = env.step(
                selection.environment_action())
            budget_reached = (
                max_environment_steps is not None
                and total_steps + 1 >= int(max_environment_steps))
            if budget_reached and not terminated:
                truncated = True
                info = dict(info)
                info["budget_truncated"] = True
            episode_transitions.append(GoalTransition(
                observation=obs,
                action=selection.discrete_action,
                action_parameters=selection.all_parameters,
                reward=reward,
                next_observation=next_obs,
                terminated=terminated,
                truncated=truncated,
                info=info,
                phase=source_phase,
            ))
            episode_return += reward
            total_steps += 1
            obs = next_obs
            final_info = info
            if terminated or truncated:
                break

        her_counts = {
            "her_relabel_count": 0,
            "cross_phase_goal_count_filtered": 0,
        }
        if isinstance(replay, ReplayBuffer):
            for transition in episode_transitions:
                replay.add(
                    state=encoder(transition.observation),
                    action=transition.action,
                    action_parameters=transition.action_parameters,
                    reward=transition.reward,
                    next_state=encoder(transition.next_observation),
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                )
        else:
            her_counts.update(replay.add_episode(
                episode_transitions,
                env.compute_reward,
                goal_validator=env.observation_space["desired_goal"].contains,
            ))

        replay_size = len(replay)
        eligible_steps = max(
            0,
            total_steps - max(steps_before_episode, int(training["learning_starts"])),
        )
        update_credit += eligible_steps
        update_multiplier = 0
        if replay_size >= int(training["batch_size"]):
            if training.get("update_schedule"):
                updates, update_multiplier = paper_update_count(
                    training,
                    episode_length=len(episode_transitions),
                    eligible_steps=update_credit,
                )
                update_credit = 0
            else:
                update_every = int(training.get("update_every", 1))
                updates, update_multiplier = paper_update_count(
                    training,
                    episode_length=len(episode_transitions),
                    eligible_steps=update_credit,
                )
                update_credit %= update_every
            for _ in range(updates):
                batch = replay.sample(int(training["batch_size"]), device=agent.device)
                losses = agent.update(batch)
                episode_losses.append(losses)
                last_losses = losses
                total_updates += 1

        success = float(bool(final_info.get("is_success", False)))
        completed_episodes += 1
        total_successes += int(success)
        success_window.append(success)
        record: dict[str, Any] = {
            "episode": episode_index,
            "environment_steps": total_steps,
            "updates": total_updates,
            "return": float(episode_return),
            "length": len(episode_transitions),
            "success": success,
            "success_rate_window": float(np.mean(success_window)),
            "success_window_size": success_window_size,
            "success_rate_run": total_successes / episode_index,
            "out_of_bounds": float(bool(final_info.get("out_of_bounds", False))),
            "boundary_hit": float(any(
                bool(t.info.get("boundary_hit", False)) for t in episode_transitions)),
            "truncated": float(episode_transitions[-1].truncated),
            "relay_reached": float(any(
                int(t.info.get("phase", t.phase)) > 0 for t in episode_transitions)),
            "epsilon": epsilon,
            "parameter_noise_std": noise,
            "replay_size": replay_size,
            "update_multiplier": update_multiplier,
            **her_counts,
        }
        if episode_losses:
            for name in ("q_loss", "parameter_actor_loss", "mean_q", "mean_target_q"):
                record[name] = float(np.mean([loss[name] for loss in episode_losses]))
        current_rate = float(np.mean(success_window))
        episode_bar.set_postfix({
            "success": f"{current_rate:.3f}",
            "return": f"{episode_return:.0f}",
            "steps": total_steps,
            "updates": total_updates,
            "replay": replay_size,
            "eps": f"{epsilon:.3f}",
        }, refresh=False)
        if current_rate > best_success_rate:
            best_success_rate = current_rate
            save_full_checkpoint(output_dir / "best.pt")
        validation_every = int(training.get("validation_every", 0))
        validation_episodes = int(training.get("validation_episodes", 0))
        if (
            validation_every > 0
            and validation_episodes > 0
            and episode_index % validation_every == 0
        ):
            validation = deterministic_validation(
                config,
                agent,
                encoder,
                episodes=validation_episodes,
                seed=int(training.get("validation_seed", 100_000 + seed)),
            )
            record.update(validation)
            score = validation["validation_success_rate"]
            mean_return = validation["validation_mean_return"]
            if (
                score > best_validation_success
                or (score == best_validation_success and mean_return > best_validation_return)
            ):
                best_validation_success = score
                best_validation_return = mean_return
                save_full_checkpoint(output_dir / "best_eval.pt")
        append_jsonl(metrics_path, record)
        checkpoint_every = int(training.get("checkpoint_every", 100))
        if checkpoint_every > 0 and episode_index % checkpoint_every == 0:
            save_full_checkpoint(output_dir / f"checkpoint_{episode_index}.pt")

    save_full_checkpoint(output_dir / "last.pt")
    env.close()
    summary = {
        "algorithm": config["experiment"]["algorithm"],
        "environment": config["experiment"]["environment"],
        "episodes": completed_episodes,
        "environment_steps": total_steps,
        "updates": total_updates,
        "success_rate_run": total_successes / completed_episodes,
        "replay_size": len(replay),
        "output_dir": str(output_dir.resolve()),
        "resumed_from": str(Path(resume_from).resolve()) if resume_from else None,
        **{name: last_losses.get(name) for name in ("q_loss", "parameter_actor_loss")},
    }
    print(json.dumps(summary, indent=2))
    return summary


# ---- Evaluation -------------------------------------------------------------

def checkpoint_config_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).resolve().parent / "config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No config.json beside checkpoint; pass --config explicitly: {path}")
    return path


def load_evaluation_config(
    checkpoint: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    return load_config(config_path or checkpoint_config_path(checkpoint))


def save_trajectory(path: Path, transitions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not transitions:
        raise ValueError("Cannot save an empty trajectory")
    keys = transitions[0].keys()
    episode = {
        key: np.asarray([transition[key] for transition in transitions])
        for key in keys
    }
    np.savez_compressed(path, **episode)


def evaluate_checkpoint(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    *,
    episodes: int = 100,
    seed: int = 10_000,
    output_dir: str | Path | None = None,
    save_trajectories: bool = False,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    config = dict(config)
    validate_config(config)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    env = make_environment(config)
    obs, _ = env.reset(seed=seed)
    encoder = make_observation_encoder(config, env)
    parameter_sizes = parameter_sizes_for_environment(env)
    agent = make_agent(config, encoder.output_dim, parameter_sizes)
    agent.load(checkpoint)
    agent.parameter_actor.eval()
    agent.q_network.eval()

    output_dir = Path(output_dir) if output_dir is not None else checkpoint.parent / "evaluation"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    # Evaluation directories are immutable per invocation from the caller's
    # perspective; overwrite stale episode records rather than append duplicates.
    episodes_path.write_text("", encoding="utf-8")
    trajectory_dir = output_dir / "trajectories"

    records: list[dict[str, Any]] = []
    eval_bar = tqdm(range(1, episodes + 1), desc=f"Eval {config['experiment']['environment']}/{config['experiment']['algorithm']}", unit="ep", dynamic_ncols=True)
    for episode_index in eval_bar:
        if episode_index > 1:
            obs, _ = env.reset()
        episode_return = 0.0
        length = 0
        relay_reached = False
        boundary_hit = False
        phase_switch_step = -1
        final_info: dict[str, Any] = {}
        trajectory: list[dict[str, Any]] = []
        while True:
            source_phase = env.current_phase
            selection = agent.select_action(
                encoder(obs), epsilon=0.0, parameter_noise_std=0.0)
            next_obs, reward, terminated, truncated, info = env.step(
                selection.environment_action())
            length += 1
            episode_return += reward
            if env.current_phase > source_phase and phase_switch_step < 0:
                phase_switch_step = length
            relay_reached = relay_reached or env.current_phase > 0
            boundary_hit = boundary_hit or bool(info.get("boundary_hit", False))
            if save_trajectories:
                trajectory.append({
                    "obs": np.asarray(obs["observation"], np.float32),
                    "action_discrete": np.int32(selection.discrete_action),
                    "action_parameter": np.float32(
                        selection.selected_parameter[0]
                        if selection.selected_parameter.size else 0.0),
                    "all_action_parameters": np.asarray(selection.all_parameters, np.float32),
                    "reward": np.float32(reward),
                    "next_obs": np.asarray(next_obs["observation"], np.float32),
                    "goal": np.asarray(obs["desired_goal"], np.float32),
                    "next_goal": np.asarray(next_obs["desired_goal"], np.float32),
                    "achieved_goal": np.asarray(obs["achieved_goal"], np.float32),
                    "next_achieved_goal": np.asarray(next_obs["achieved_goal"], np.float32),
                    "phase": np.int32(source_phase),
                    "next_phase": np.int32(env.current_phase),
                    "terminated": np.bool_(terminated),
                    "truncated": np.bool_(truncated),
                })
            obs = next_obs
            final_info = info
            if terminated or truncated:
                break
        record = {
            "episode": episode_index,
            "return": float(episode_return),
            "length": length,
            "success": float(bool(final_info.get("is_success", False))),
            "out_of_bounds": float(bool(final_info.get("out_of_bounds", False))),
            "boundary_hit": float(boundary_hit),
            "truncated": float(bool(truncated)),
            "relay_reached": float(relay_reached),
            "phase_switch_step": phase_switch_step,
        }
        records.append(record)
        eval_bar.set_postfix({"success": f"{np.mean([r['success'] for r in records]):.3f}", "return": f"{episode_return:.0f}"}, refresh=False)
        append_jsonl(episodes_path, record)
        if save_trajectories:
            save_trajectory(trajectory_dir / f"episode_{episode_index:05d}.npz", trajectory)

    env.close()
    switched = [record["phase_switch_step"] for record in records if record["phase_switch_step"] >= 0]
    summary = {
        "algorithm": config["experiment"]["algorithm"],
        "environment": config["experiment"]["environment"],
        "checkpoint": str(checkpoint),
        "evaluation_seed": seed,
        "episodes": episodes,
        "mean_return": float(np.mean([record["return"] for record in records])),
        "std_return": float(np.std([record["return"] for record in records])),
        "mean_length": float(np.mean([record["length"] for record in records])),
        "success_rate": float(np.mean([record["success"] for record in records])),
        "out_of_bounds_rate": float(np.mean([record["out_of_bounds"] for record in records])),
        "boundary_hit_rate": float(np.mean([record["boundary_hit"] for record in records])),
        "truncation_rate": float(np.mean([record["truncated"] for record in records])),
        "relay_reached_rate": float(np.mean([record["relay_reached"] for record in records])),
        "mean_phase_switch_step": float(np.mean(switched)) if switched else None,
        "trajectories_saved": episodes if save_trajectories else 0,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "evaluation_config.json").write_text(
        json.dumps({
            "episodes": episodes,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "source_config": config.get("_config_path"),
        }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


# ---- Plotting ---------------------------------------------------------------
def read_runs(root: Path) -> list[dict]:
    runs = []
    for path in root.rglob("metrics.jsonl"):
        config_path = path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if metrics:
            runs.append({"config": config, "metrics": metrics})
    return runs


def plot_learning_curves(root: Path, output: Path) -> list[Path]:
    runs = read_runs(root)
    created: list[Path] = []
    for environment in sorted({r["config"]["experiment"]["environment"] for r in runs}):
        env_runs = [r for r in runs if r["config"]["experiment"]["environment"] == environment]
        fig, ax = plt.subplots(figsize=(8, 5))
        for algorithm in sorted({r["config"]["experiment"]["algorithm"] for r in env_runs}):
            group = [r for r in env_runs if r["config"]["experiment"]["algorithm"] == algorithm]
            max_step = min(r["metrics"][-1]["environment_steps"] for r in group)
            grid = np.linspace(0, max_step, 101)
            curves = []
            for run in group:
                x = np.asarray([m["environment_steps"] for m in run["metrics"]], float)
                y = np.asarray([m.get("success_rate_window", m.get("success_rate_100", 0.0)) for m in run["metrics"]], float)
                curves.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
            values = np.asarray(curves)
            mean, std = values.mean(0), values.std(0)
            ax.plot(grid, mean, label=algorithm.upper().replace("_", "-"))
            ax.fill_between(grid, mean - std, mean + std, alpha=0.18)
        ax.set(title=f"{environment.title()} navigation", xlabel="Environment steps", ylabel="Training success rate (paper rolling window)", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output / f"{environment}_learning_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)
    return created


def plot_evaluation(root: Path, output: Path) -> Path | None:
    path = root / "aggregate_summary.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
    environments = sorted({row["environment"] for row in rows})
    algorithms = ["pdqn", "mpdqn", "her_pdqn", "her_mpdqn"]
    x = np.arange(len(algorithms), dtype=float)
    width = 0.8 / max(1, len(environments))
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, environment in enumerate(environments):
        mapping = {row["algorithm"]: row for row in rows if row["environment"] == environment}
        means = [mapping.get(a, {}).get("success_rate_mean", 0.0) for a in algorithms]
        stds = [mapping.get(a, {}).get("success_rate_std", 0.0) for a in algorithms]
        ax.bar(x + (index - (len(environments) - 1) / 2) * width, means, width, yerr=stds, capsize=3, label=environment.title())
    ax.set(xticks=x, xticklabels=[a.upper().replace("_", "-") for a in algorithms], ylabel="Evaluation success rate", ylim=(0, 1.05), title="Fixed-step baseline comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = output / "evaluation_success_rate.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def plot_horizon(root: Path, output: Path) -> Path | None:
    path = root / "horizon_summary.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
    relays = np.asarray([row["num_relays"] for row in rows])
    success = np.asarray([row["success_rate_mean"] for row in rows])
    success_std = np.asarray([row["success_rate_std"] for row in rows])
    reached = np.asarray([row["relay_reached_rate_mean"] for row in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(relays, success, yerr=success_std, marker="o", capsize=4, label="Final success")
    ax.plot(relays, reached, marker="s", label="Reached at least one relay")
    ax.set(xticks=relays, xlabel="Number of relays", ylabel="Evaluation rate", ylim=(-0.02, 1.02), title="HER-MPDQN horizon scaling")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = output / "horizon_scaling.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


# ---- Baseline suite ---------------------------------------------------------
def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for environment in sorted({r["environment"] for r in records}):
        for algorithm in ALGORITHMS:
            group = [r for r in records if r["environment"] == environment and r["algorithm"] == algorithm]
            if not group:
                continue
            row: dict[str, Any] = {
                "environment": environment,
                "algorithm": algorithm,
                "num_seeds": len(group),
                "seeds": [r["seed"] for r in group],
            }
            for key in (
                "success_rate", "mean_return", "mean_length", "out_of_bounds_rate", "boundary_hit_rate",
                "relay_reached_rate", "training_success_rate", "environment_steps",
                "q_loss", "parameter_actor_loss", "her_relabel_count",
                "best_eval_success_rate", "best_eval_mean_return",
            ):
                values = [float(r[key]) for r in group if r.get(key) is not None]
                row[f"{key}_mean"] = float(np.mean(values)) if values else None
                row[f"{key}_std"] = float(np.std(values)) if values else None
            rows.append(row)
    return rows


def metric_totals(metrics_path: Path) -> dict[str, float]:
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        return {"her_relabel_count": 0.0}
    return {
        "her_relabel_count": float(sum(r.get("her_relabel_count", 0) for r in records)),
        "q_loss": records[-1].get("q_loss"),
        "parameter_actor_loss": records[-1].get("parameter_actor_loss"),
        "training_success_rate": records[-1].get("success_rate_run"),
        "environment_steps": records[-1].get("environment_steps"),
    }


def latest_training_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for path in run_dir.glob("checkpoint_*.pt"):
        try:
            int(path.stem.split("_")[-1])
            candidates.append(path)
        except ValueError:
            continue
    last = run_dir / "last.pt"
    if last.exists():
        candidates.append(last)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one independent seed/run inside its own process."""
    import torch

    torch.set_num_threads(int(job.get("worker_threads", 1)))
    environment = job["environment"]
    algorithm = job["algorithm"]
    seed = int(job["seed"])
    steps = job.get("steps")
    steps = None if steps is None else int(steps)
    eval_episodes = int(job["eval_episodes"])
    reuse_existing = bool(job["reuse_existing"])
    run_dir = Path(job["run_dir"])
    checkpoint = run_dir / "last.pt"
    evaluation_path = run_dir / "evaluation" / "summary.json"
    if reuse_existing and checkpoint.exists() and evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    else:
        config = get_config(environment, algorithm, seed=seed, output_dir=str(run_dir))
        resume_from = latest_training_checkpoint(run_dir) if reuse_existing else None
        training_override = {} if steps is None else {"max_environment_steps": steps}
        train(config, overrides={
            "experiment": {"seed": seed, "output_dir": str(run_dir)},
            "training": training_override,
        }, resume_from=resume_from)
        evaluation = evaluate_checkpoint(
            load_config(run_dir / "config.json"), checkpoint,
            episodes=eval_episodes, seed=100_000 + seed,
            output_dir=run_dir / "evaluation",
        )
    best_evaluation = None
    best_checkpoint = run_dir / "best_eval.pt"
    best_evaluation_path = run_dir / "evaluation_best" / "summary.json"
    if best_checkpoint.exists():
        if reuse_existing and best_evaluation_path.exists():
            best_evaluation = json.loads(best_evaluation_path.read_text(encoding="utf-8"))
        else:
            best_evaluation = evaluate_checkpoint(
                load_config(run_dir / "config.json"), best_checkpoint,
                episodes=eval_episodes, seed=100_000 + seed,
                output_dir=run_dir / "evaluation_best",
            )
    totals = metric_totals(run_dir / "metrics.jsonl")
    return {
        "environment": environment,
        "algorithm": algorithm,
        "seed": seed,
        "success_rate": evaluation["success_rate"],
        "mean_return": evaluation["mean_return"],
        "mean_length": evaluation["mean_length"],
        "out_of_bounds_rate": evaluation["out_of_bounds_rate"],
        "boundary_hit_rate": evaluation.get("boundary_hit_rate", 0.0),
        "relay_reached_rate": evaluation["relay_reached_rate"],
        "training_success_rate": totals.get("training_success_rate"),
        "environment_steps": totals.get("environment_steps", steps),
        "q_loss": totals.get("q_loss"),
        "parameter_actor_loss": totals.get("parameter_actor_loss"),
        "her_relabel_count": totals["her_relabel_count"],
        "best_eval_success_rate": (
            best_evaluation["success_rate"] if best_evaluation is not None else None),
        "best_eval_mean_return": (
            best_evaluation["mean_return"] if best_evaluation is not None else None),
        "run_dir": str(run_dir),
    }


def run_suite(
    *, environments: list[str], algorithms: list[str], seeds: list[int], steps: int | None,
    eval_episodes: int, output_root: Path, reuse_existing: bool = False,
    workers: int = 1, worker_threads: int = 1,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if worker_threads <= 0:
        raise ValueError("worker_threads must be positive")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for environment in environments:
        for algorithm in algorithms:
            for seed in seeds:
                run_dir = output_root / environment / algorithm / f"seed_{seed}"
                jobs.append({
                    "environment": environment, "algorithm": algorithm,
                    "seed": seed, "steps": steps, "eval_episodes": eval_episodes,
                    "reuse_existing": reuse_existing, "run_dir": str(run_dir),
                    "worker_threads": worker_threads,
                })

    records: list[dict[str, Any]] = []

    def persist() -> None:
        ordered = sorted(records, key=lambda r: (
            environments.index(r["environment"]), algorithms.index(r["algorithm"]), r["seed"]))
        (output_root / "per_run.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered),
            encoding="utf-8")

    if workers == 1:
        for job in tqdm(jobs, desc="Baseline runs", unit="run", dynamic_ncols=True):
            records.append(run_one(job))
            persist()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, job): job for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Baseline runs", unit="run", dynamic_ncols=True):
                records.append(future.result())
                persist()

    aggregated = aggregate(records)
    result = {
        "fixed_environment_steps": steps,
        "evaluation_episodes": eval_episodes,
        "num_runs": len(records),
        "runs": records,
        "aggregate": aggregated,
    }
    (output_root / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if aggregated:
        with (output_root / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregated[0]))
            writer.writeheader()
            writer.writerows(aggregated)
    return result


# ---- Non-paper multi-relay extension ---------------------------------------
def run_horizons(*, relay_counts: list[int], seeds: list[int], steps: int,
                 eval_episodes: int, output_root: Path) -> dict:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    base = get_config("multi_relay", "her_mpdqn")
    for num_relays in relay_counts:
        for seed in seeds:
            run_dir = output_root / f"relays_{num_relays}" / f"seed_{seed}"
            training = train(base, overrides={
                "experiment": {"seed": seed, "output_dir": str(run_dir)},
                "env": {
                    "num_relays": num_relays,
                    "max_episode_steps": max(100, 100 * (num_relays + 1)),
                },
                "training": {
                    "max_environment_steps": steps,
                    "learning_starts": min(2_000, max(128, steps // 3)),
                },
            })
            evaluation = evaluate_checkpoint(
                load_config(run_dir / "config.json"), run_dir / "last.pt",
                episodes=eval_episodes, seed=200_000 + seed,
                output_dir=run_dir / "evaluation",
            )
            totals = metric_totals(run_dir / "metrics.jsonl")
            record = {
                "num_relays": num_relays,
                "num_phases": num_relays + 1,
                "seed": seed,
                "environment_steps": training["environment_steps"],
                "success_rate": evaluation["success_rate"],
                "mean_return": evaluation["mean_return"],
                "mean_length": evaluation["mean_length"],
                "relay_reached_rate": evaluation["relay_reached_rate"],
                "q_loss": totals.get("q_loss"),
                "parameter_actor_loss": totals.get("parameter_actor_loss"),
                "her_relabel_count": totals["her_relabel_count"],
                "run_dir": str(run_dir),
            }
            records.append(record)
            (output_root / "per_run.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
                encoding="utf-8")
    aggregate = []
    for num_relays in relay_counts:
        group = [row for row in records if row["num_relays"] == num_relays]
        result = {"num_relays": num_relays, "num_phases": num_relays + 1, "num_seeds": len(group)}
        for key in ("success_rate", "mean_return", "mean_length", "relay_reached_rate"):
            values = np.asarray([row[key] for row in group], dtype=float)
            result[f"{key}_mean"] = float(values.mean())
            result[f"{key}_std"] = float(values.std())
        aggregate.append(result)
    summary = {"steps": steps, "eval_episodes": eval_episodes, "runs": records, "aggregate": aggregate}
    (output_root / "horizon_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_root / "horizon_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    return summary


# ---- Optional direct pilot --------------------------------------------------

ALGORITHMS = ("pdqn", "mpdqn", "her_pdqn", "her_mpdqn")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def window_mean(records, key, start, stop):
    values = [record[key] for record in records[start:stop] if key in record]
    return float(np.mean(values)) if values else None


def run_direct_pilot(*, steps: int = 5000, eval_episodes: int = 30, seed: int = 0,
                     output_root: Path = Path("outputs/pilot/direct")) -> dict[str, Any]:
    output_root = output_root if output_root.is_absolute() else ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for algorithm in ALGORITHMS:
        output_dir = output_root / algorithm / f"seed_{seed}"
        config = get_config("direct", algorithm, seed=seed, output_dir=str(output_dir))
        training_summary = train(config, overrides={
            "experiment": {"seed": seed, "output_dir": str(output_dir)},
            "training": {"max_environment_steps": steps, "checkpoint_every": 0},
        })
        records = read_jsonl(output_dir / "metrics.jsonl")
        window = min(10, len(records))
        evaluation = evaluate_checkpoint(
            load_evaluation_config(output_dir / "last.pt"), output_dir / "last.pt",
            episodes=eval_episodes, seed=10_000 + seed,
            output_dir=output_dir / "evaluation_last", save_trajectories=False)
        results.append({
            "algorithm": algorithm, "seed": seed,
            "training_steps": training_summary["environment_steps"],
            "early_success_rate_10": window_mean(records, "success", 0, window),
            "late_success_rate_10": window_mean(records, "success", -window, None),
            "evaluation_success_rate": evaluation["success_rate"],
            "evaluation_mean_return": evaluation["mean_return"],
        })
    artifact = {"environment": "direct", "steps": steps,
                "evaluation_episodes": eval_episodes, "seed": seed, "results": results}
    (output_root / "pilot_summary.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


class Main:
    """Unified controller for training, evaluation, baseline suites and optional experiments."""

    def __init__(self, *, run_all_baselines: bool = True, run_smoke: bool = False,
                 run_direct_pilot: bool = False, run_horizon: bool = False):
        self.run_all_baselines_enabled = bool(run_all_baselines)
        self.run_smoke_enabled = bool(run_smoke)
        self.run_direct_pilot_enabled = bool(run_direct_pilot)
        self.run_horizon_enabled = bool(run_horizon)

    def smoke(self, environments=("direct", "relay"), episodes=3):
        for environment in environments:
            for algorithm in ALGORITHMS:
                train(get_config(environment, algorithm), overrides={
                    "experiment": {"output_dir": f"outputs/smoke/{environment}_{algorithm}"},
                    "env": {"max_episode_steps": 8},
                    "agent": {"hidden_sizes": [16], "device": "cpu"},
                    "replay": {"capacity": 512},
                    "training": {"episodes": episodes, "learning_starts": 0, "batch_size": 4,
                                 "update_every": 2, "gradient_steps": 1, "checkpoint_every": 0},
                    "exploration": {"decay_steps": 16},
                })

    def execute(self, *, task=2, algorithms=ALGORITHMS, seeds=(0,1,2,3,4,5,6),
                steps=None, eval_episodes=1000, output_root="outputs/paper_reproduction_7seed",
                workers=1, worker_threads=1, reuse_existing=False, relay_counts=(0,1,2,4,8)):
        if task not in (1, 2):
            raise ValueError("task must be 1 (Direct Navigation) or 2 (Relay Navigation)")
        environments = ("direct",) if task == 1 else ("relay",)
        results = {}
        if self.run_smoke_enabled:
            self.smoke(environments=environments)
        if self.run_direct_pilot_enabled:
            results["pilot"] = run_direct_pilot(
                steps=5000 if steps is None else int(steps),
                eval_episodes=min(eval_episodes, 30),
                seed=int(seeds[0]),
                output_root=Path(output_root) / "pilot",
            )
        if self.run_all_baselines_enabled:
            results["baselines"] = run_suite(
                environments=list(environments), algorithms=list(algorithms), seeds=list(seeds),
                steps=steps, eval_episodes=eval_episodes, output_root=Path(output_root),
                reuse_existing=reuse_existing, workers=workers, worker_threads=worker_threads)
        if self.run_horizon_enabled:
            results["horizon"] = run_horizons(
                relay_counts=list(relay_counts), seeds=list(seeds),
                steps=5000 if steps is None else int(steps), eval_episodes=min(eval_episodes, 30),
                output_root=Path(output_root) / "horizon")
        return results


def main():
    parser = argparse.ArgumentParser(description="Unified HER-MPDQN experiment runner")
    parser.add_argument("--task", type=int, choices=(1, 2), default=2,
                        help="Paper task: 1=Direct Navigation, 2=Relay Navigation (default)")
    parser.add_argument("--run-all-baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-direct-pilot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-horizon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--output-root", default="outputs/paper_reproduction_7seed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--relay-counts", default="0,1,2,4,8")
    args = parser.parse_args()
    controller = Main(
        run_all_baselines=args.run_all_baselines, run_smoke=args.run_smoke,
        run_direct_pilot=args.run_direct_pilot, run_horizon=args.run_horizon)
    result = controller.execute(
        task=args.task, algorithms=parse_csv(args.algorithms),
        seeds=parse_csv(args.seeds, int), steps=args.steps, eval_episodes=args.eval_episodes,
        output_root=args.output_root, workers=args.workers, worker_threads=args.worker_threads,
        reuse_existing=args.reuse_existing, relay_counts=parse_csv(args.relay_counts, int))
    print(json.dumps({k: (v if isinstance(v, str) else "completed") for k,v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
