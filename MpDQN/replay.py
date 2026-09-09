"""Replay storage and hindsight relabeling for off-policy agents.

Single-file merge of goal_relabeling.py, replay_buffer.py, her_buffer.py,
and the package exports previously declared in __init__.py.
"""

from __future__ import annotations


# === Goal relabeling ===
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


GoalObservation = Mapping[str, np.ndarray]
ComputeReward = Callable[[np.ndarray, np.ndarray, Any], float | np.ndarray]
GoalValidator = Callable[[np.ndarray], bool]


def copy_goal_observation(observation: GoalObservation) -> dict[str, np.ndarray]:
    required = ("observation", "achieved_goal", "desired_goal")
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"Goal observation is missing keys: {missing}")
    copied = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in observation.items()
    }
    if copied["achieved_goal"].shape != copied["desired_goal"].shape:
        raise ValueError("achieved_goal and desired_goal must have matching shapes")
    for key, value in copied.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Observation field {key} contains non-finite values")
    return copied


@dataclass(frozen=True)
class GoalTransition:
    """One environment step retained in episode order for HER.

    ``phase`` is the task phase at the source state, before executing the
    action. This convention keeps the relay-acquiring CATCH transition in
    phase 0 even though its next observation is already in phase 1.
    """

    observation: GoalObservation
    action: int
    action_parameters: np.ndarray
    reward: float
    next_observation: GoalObservation
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    phase: int = 0

    def __post_init__(self) -> None:
        observation = copy_goal_observation(self.observation)
        next_observation = copy_goal_observation(self.next_observation)
        if observation["achieved_goal"].shape != next_observation["achieved_goal"].shape:
            raise ValueError("Goal shape must remain constant within a transition")
        parameters = np.asarray(self.action_parameters, dtype=np.float32).copy()
        if parameters.ndim != 1 or not np.all(np.isfinite(parameters)):
            raise ValueError("action_parameters must be a finite one-dimensional vector")
        if not np.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if self.terminated and self.truncated:
            raise ValueError("A transition cannot be both terminated and truncated")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "action_parameters", parameters)
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "info", dict(self.info))
        object.__setattr__(self, "phase", int(self.phase))


@dataclass(frozen=True)
class RelabeledTransition:
    transition: GoalTransition
    source_index: int
    future_index: int
    relabeled_goal: np.ndarray


class FutureGoalRelabeler:
    """Sample up to ``her_k`` unique achieved goals from t+1..T."""

    def __init__(self, her_k: int = 4, seed: int = 0) -> None:
        if her_k < 0:
            raise ValueError("her_k must be non-negative")
        self.her_k = int(her_k)
        self.rng = np.random.default_rng(seed)

    def _future_candidates(
        self,
        episode: Sequence[GoalTransition],
        source_index: int,
        goal_validator: GoalValidator | None,
    ) -> list[int]:
        candidates = []
        # transition j ends at state s_(j+1), so j >= source_index is future
        # relative to the source state s_t.
        for future_index in range(source_index, len(episode)):
            goal = episode[future_index].next_observation["achieved_goal"]
            if goal_validator is None or bool(goal_validator(goal)):
                candidates.append(future_index)
        return candidates

    def relabel_episode(
        self,
        episode: Sequence[GoalTransition],
        compute_reward: ComputeReward,
        *,
        goal_validator: GoalValidator | None = None,
    ) -> list[RelabeledTransition]:
        if not episode or self.her_k == 0:
            return []
        relabeled: list[RelabeledTransition] = []
        for source_index, source in enumerate(episode):
            candidates = self._future_candidates(episode, source_index, goal_validator)
            count = min(self.her_k, len(candidates))
            if count == 0:
                continue
            sampled = self.rng.choice(candidates, size=count, replace=False)
            for future_index in np.atleast_1d(sampled):
                future_index = int(future_index)
                goal = np.asarray(
                    episode[future_index].next_observation["achieved_goal"],
                    dtype=np.float32,
                ).copy()
                observation = copy_goal_observation(source.observation)
                next_observation = copy_goal_observation(source.next_observation)
                observation["desired_goal"] = goal.copy()
                next_observation["desired_goal"] = goal.copy()
                relabel_info = {
                    **source.info,
                    "is_her": True,
                    "her_source_action": int(source.action),
                    "her_source_phase": int(source.phase),
                    "her_achieved_goal_before": np.asarray(
                        source.observation["achieved_goal"], dtype=np.float32).copy(),
                }
                reward_value = compute_reward(
                    next_observation["achieved_goal"], goal, relabel_info)
                reward_array = np.asarray(reward_value, dtype=np.float32)
                if reward_array.shape != ():
                    raise ValueError("compute_reward must return a scalar for one transition")
                reward = float(reward_array)
                if not np.isfinite(reward):
                    raise ValueError("compute_reward returned a non-finite reward")
                virtual_success = reward == 0.0
                transition = GoalTransition(
                    observation=observation,
                    action=source.action,
                    action_parameters=source.action_parameters,
                    reward=reward,
                    next_observation=next_observation,
                    # Sparse goal completion is terminal. A time-limit
                    # truncation remains a truncation only when the relabeled
                    # goal was not achieved at this transition.
                    terminated=virtual_success,
                    truncated=source.truncated and not virtual_success,
                    info={**relabel_info, "is_success": virtual_success},
                    phase=source.phase,
                )
                relabeled.append(RelabeledTransition(
                    transition=transition,
                    source_index=source_index,
                    future_index=future_index,
                    relabeled_goal=goal,
                ))
        return relabeled


class PhaseAwareFutureGoalRelabeler(FutureGoalRelabeler):
    """GSM reconstruction: future goals must come from the source phase."""

    def __init__(self, her_k: int = 4, seed: int = 0) -> None:
        super().__init__(her_k=her_k, seed=seed)
        self.last_cross_phase_filtered = 0
        self.total_cross_phase_filtered = 0

    @staticmethod
    def _validate_phases(episode: Sequence[GoalTransition]) -> None:
        phases = [transition.phase for transition in episode]
        if any(phase < 0 for phase in phases):
            raise ValueError("Episode phases must be non-negative")
        for previous, current in zip(phases, phases[1:]):
            if current < previous:
                raise ValueError("Episode phases must be monotonically non-decreasing")
            if current > previous + 1:
                raise ValueError("Episode phases cannot skip a stage")

    def _future_candidates(
        self,
        episode: Sequence[GoalTransition],
        source_index: int,
        goal_validator: GoalValidator | None,
    ) -> list[int]:
        valid_future = super()._future_candidates(
            episode, source_index, goal_validator)
        source_phase = episode[source_index].phase
        same_phase = [
            index for index in valid_future
            if episode[index].phase == source_phase
        ]
        filtered = len(valid_future) - len(same_phase)
        self.last_cross_phase_filtered += filtered
        self.total_cross_phase_filtered += filtered
        return same_phase

    def relabel_episode(
        self,
        episode: Sequence[GoalTransition],
        compute_reward: ComputeReward,
        *,
        goal_validator: GoalValidator | None = None,
    ) -> list[RelabeledTransition]:
        self.last_cross_phase_filtered = 0
        self._validate_phases(episode)
        return super().relabel_episode(
            episode, compute_reward, goal_validator=goal_validator)

# === Replay buffer ===
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents import ActionSelection, TransitionBatch


@dataclass(frozen=True)
class ReplayBufferConfig:
    capacity: int
    state_dim: int
    parameter_dim: int
    num_actions: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.state_dim <= 0 or self.parameter_dim <= 0 or self.num_actions <= 0:
            raise ValueError("Replay dimensions must be positive")


class ReplayBuffer:
    """Preallocated circular replay memory.

    ``terminated`` and ``truncated`` are stored separately. By default samples
    bootstrap across time-limit truncation but not true MDP termination, which
    matches Gymnasium semantics and prevents a time limit from becoming a fake
    terminal state.
    """

    FORMAT_VERSION = 1

    def __init__(self, config: ReplayBufferConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Zero initialization keeps unused capacity deterministic and prevents
        # checkpointing uninitialized process-memory bytes.
        self.states = np.zeros((config.capacity, config.state_dim), dtype=np.float32)
        self.actions = np.zeros(config.capacity, dtype=np.int64)
        self.action_parameters = np.zeros(
            (config.capacity, config.parameter_dim), dtype=np.float32)
        self.rewards = np.zeros(config.capacity, dtype=np.float32)
        self.next_states = np.zeros((config.capacity, config.state_dim), dtype=np.float32)
        self.terminated = np.zeros(config.capacity, dtype=np.bool_)
        self.truncated = np.zeros(config.capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    @property
    def full(self) -> bool:
        return self.size == self.config.capacity

    def _vector(self, value: np.ndarray, dimension: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (dimension,):
            raise ValueError(f"{name} must have shape {(dimension,)}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        return array

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        action_parameters: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> int:
        """Add one transition and return its physical storage index."""

        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer discrete-action index")
        action = int(action)
        if not 0 <= action < self.config.num_actions:
            raise ValueError(f"action must lie in [0, {self.config.num_actions})")
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if bool(terminated) and bool(truncated):
            raise ValueError("A transition cannot be both terminated and truncated")

        index = self.position
        self.states[index] = self._vector(state, self.config.state_dim, "state")
        self.actions[index] = action
        self.action_parameters[index] = self._vector(
            action_parameters, self.config.parameter_dim, "action_parameters")
        self.rewards[index] = reward
        self.next_states[index] = self._vector(
            next_state, self.config.state_dim, "next_state")
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)

        self.position = (self.position + 1) % self.config.capacity
        self.size = min(self.size + 1, self.config.capacity)
        self.total_added += 1
        return index

    def add_selection(
        self,
        *,
        state: np.ndarray,
        selection: ActionSelection,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> int:
        """Add the complete parameter vector returned by an agent policy."""

        return self.add(
            state=state,
            action=selection.discrete_action,
            action_parameters=selection.all_parameters,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer containing {self.size}")
        return self.rng.choice(self.size, size=batch_size, replace=False)

    def batch_from_indices(
        self,
        indices: np.ndarray,
        *,
        device: torch.device | str = "cpu",
        bootstrap_on_truncation: bool = True,
    ) -> TransitionBatch:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("indices must not be empty")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Replay indices are outside the currently stored range")
        dones = self.terminated[indices].copy()
        if not bootstrap_on_truncation:
            dones = np.logical_or(dones, self.truncated[indices])
        return TransitionBatch.from_numpy(
            states=self.states[indices],
            actions=self.actions[indices],
            action_parameters=self.action_parameters[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=dones.astype(np.float32),
            device=device,
        )

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        bootstrap_on_truncation: bool = True,
    ) -> TransitionBatch:
        return self.batch_from_indices(
            self.sample_indices(batch_size),
            device=device,
            bootstrap_on_truncation=bootstrap_on_truncation,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a complete state, including circular position and RNG state."""

        return {
            "format_version": self.FORMAT_VERSION,
            "config": asdict(self.config),
            "position": self.position,
            "size": self.size,
            "total_added": self.total_added,
            "rng_state": self.rng.bit_generator.state,
            "states": self.states.copy(),
            "actions": self.actions.copy(),
            "action_parameters": self.action_parameters.copy(),
            "rewards": self.rewards.copy(),
            "next_states": self.next_states.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["format_version"]) != self.FORMAT_VERSION:
            raise ValueError("Unsupported replay format version")
        if dict(state["config"]) != asdict(self.config):
            raise ValueError("Replay checkpoint configuration does not match this buffer")
        for name in (
            "states", "actions", "action_parameters", "rewards",
            "next_states", "terminated", "truncated"):
            source = np.asarray(state[name])
            destination = getattr(self, name)
            if source.shape != destination.shape:
                raise ValueError(f"Replay array {name} has incompatible shape")
            destination[...] = source.astype(destination.dtype, copy=False)
        self.position = int(state["position"])
        self.size = int(state["size"])
        self.total_added = int(state["total_added"])
        if not 0 <= self.position < self.config.capacity or not 0 <= self.size <= self.config.capacity:
            raise ValueError("Replay checkpoint has invalid position or size")
        self.rng.bit_generator.state = dict(state["rng_state"])

    def save(self, path: str | Path) -> None:
        """Save replay without pickle using compressed NumPy arrays plus JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        metadata = {
            key: state[key]
            for key in (
                "format_version", "config", "position", "size", "total_added", "rng_state")
        }
        arrays = {
            key: state[key]
            for key in (
                "states", "actions", "action_parameters", "rewards",
                "next_states", "terminated", "truncated")
        }
        np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            buffer = cls(ReplayBufferConfig(**metadata["config"]))
            state = dict(metadata)
            for key in (
                "states", "actions", "action_parameters", "rewards",
                "next_states", "terminated", "truncated"):
                state[key] = archive[key]
            buffer.load_state_dict(state)
        return buffer

# === HER replay integration ===
from collections.abc import Sequence
from typing import Any, Callable, Mapping

import numpy as np
import torch

from agents import TransitionBatch, goal_observation_to_vector



class HERReplayBuffer:
    """Expand complete episodes into original plus hindsight transitions."""

    def __init__(
        self,
        config: ReplayBufferConfig,
        *,
        her_k: int = 4,
        seed: int | None = None,
        relabeler: FutureGoalRelabeler | None = None,
        state_encoder: Callable[[Any], np.ndarray] = goal_observation_to_vector,
    ) -> None:
        self.replay = ReplayBuffer(config)
        if relabeler is not None and relabeler.her_k != her_k:
            raise ValueError("Provided relabeler her_k does not match buffer her_k")
        self.relabeler = relabeler or FutureGoalRelabeler(
            her_k=her_k, seed=config.seed if seed is None else seed)
        self.her_k = int(her_k)
        self.state_encoder = state_encoder
        self.total_original_added = 0
        self.total_relabelled_added = 0
        self.episodes_added = 0
        self.total_cross_phase_filtered = 0

    def __len__(self) -> int:
        return len(self.replay)

    def _add_transition(self, transition: GoalTransition) -> int:
        return self.replay.add(
            state=self.state_encoder(transition.observation),
            action=transition.action,
            action_parameters=transition.action_parameters,
            reward=transition.reward,
            next_state=self.state_encoder(transition.next_observation),
            terminated=transition.terminated,
            truncated=transition.truncated,
        )

    def add_episode(
        self,
        episode: Sequence[GoalTransition],
        compute_reward: ComputeReward,
        *,
        goal_validator: GoalValidator | None = None,
    ) -> dict[str, int]:
        if not episode:
            raise ValueError("Cannot add an empty episode")
        episode = tuple(episode)
        for transition in episode:
            self._add_transition(transition)
        relabeled = self.relabeler.relabel_episode(
            episode, compute_reward, goal_validator=goal_validator)
        for item in relabeled:
            self._add_transition(item.transition)
        original_count = len(episode)
        relabel_count = len(relabeled)
        self.total_original_added += original_count
        self.total_relabelled_added += relabel_count
        self.episodes_added += 1
        cross_phase_filtered = int(
            getattr(self.relabeler, "last_cross_phase_filtered", 0))
        self.total_cross_phase_filtered += cross_phase_filtered
        counts = {
            "original_transition_count": original_count,
            "her_relabel_count": relabel_count,
            "stored_transition_count": original_count + relabel_count,
            "total_her_relabel_count": self.total_relabelled_added,
        }
        if isinstance(self.relabeler, PhaseAwareFutureGoalRelabeler):
            counts.update({
                "cross_phase_goal_count_filtered": cross_phase_filtered,
                "total_cross_phase_goal_count_filtered": self.total_cross_phase_filtered,
            })
        return counts

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        bootstrap_on_truncation: bool = True,
    ) -> TransitionBatch:
        return self.replay.sample(
            batch_size,
            device=device,
            bootstrap_on_truncation=bootstrap_on_truncation,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return replay storage, HER counters, and both sampling RNG states."""
        return {
            "format_version": 1,
            "buffer_type": type(self).__name__,
            "her_k": self.her_k,
            "replay": self.replay.state_dict(),
            "relabeler_rng_state": self.relabeler.rng.bit_generator.state,
            "total_original_added": self.total_original_added,
            "total_relabelled_added": self.total_relabelled_added,
            "episodes_added": self.episodes_added,
            "total_cross_phase_filtered": self.total_cross_phase_filtered,
            "relabeler_total_cross_phase_filtered": int(
                getattr(self.relabeler, "total_cross_phase_filtered", 0)),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("format_version", -1)) != 1:
            raise ValueError("Unsupported HER replay format version")
        if state.get("buffer_type") != type(self).__name__:
            raise ValueError("HER replay checkpoint type does not match this buffer")
        if int(state["her_k"]) != self.her_k:
            raise ValueError("HER replay checkpoint her_k does not match this buffer")
        self.replay.load_state_dict(state["replay"])
        self.relabeler.rng.bit_generator.state = dict(state["relabeler_rng_state"])
        self.total_original_added = int(state["total_original_added"])
        self.total_relabelled_added = int(state["total_relabelled_added"])
        self.episodes_added = int(state["episodes_added"])
        self.total_cross_phase_filtered = int(state["total_cross_phase_filtered"])
        if hasattr(self.relabeler, "total_cross_phase_filtered"):
            self.relabeler.total_cross_phase_filtered = int(
                state.get("relabeler_total_cross_phase_filtered", 0))

    @property
    def metrics(self) -> dict[str, int]:
        metrics = {
            "episodes_added": self.episodes_added,
            "original_transition_count": self.total_original_added,
            "her_relabel_count": self.total_relabelled_added,
            "stored_transition_count": len(self.replay),
            "total_transitions_seen": self.replay.total_added,
        }
        if isinstance(self.relabeler, PhaseAwareFutureGoalRelabeler):
            metrics["cross_phase_goal_count_filtered"] = self.total_cross_phase_filtered
        return metrics


class PhaseAwareHERReplayBuffer(HERReplayBuffer):
    """HER replay using same-phase future goals for relay/multi-stage tasks."""

    def __init__(
        self,
        config: ReplayBufferConfig,
        *,
        her_k: int = 4,
        seed: int | None = None,
        state_encoder: Callable[[Any], np.ndarray] = goal_observation_to_vector,
    ) -> None:
        relabel_seed = config.seed if seed is None else seed
        super().__init__(
            config,
            her_k=her_k,
            seed=relabel_seed,
            relabeler=PhaseAwareFutureGoalRelabeler(
                her_k=her_k, seed=relabel_seed),
            state_encoder=state_encoder,
        )

__all__ = [
    "FutureGoalRelabeler",
    "GoalTransition",
    "HERReplayBuffer",
    "PhaseAwareFutureGoalRelabeler",
    "PhaseAwareHERReplayBuffer",
    "RelabeledTransition",
    "ReplayBuffer",
    "ReplayBufferConfig",
]
