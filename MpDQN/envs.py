"""Unified UAV environments module.

Single-file environment implementation for the paper reconstruction.
Direct and relay tasks follow the paper where specified; MultiRelay is an
explicit extension and is not part of the original benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ==============================================================================
# Merged from dynamics.py
# ==============================================================================

MOVE = 0
TURN = 1
CATCH = 2


@dataclass(frozen=True)
class DynamicsConfig:
    """Physical scaling used to turn normalized parameters into controls."""

    max_acceleration: float = 4.0
    max_turn_angle: float = math.pi / 3.0
    min_speed: float = 0.0
    max_speed: float = 40.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.max_acceleration <= 0 or self.max_turn_angle <= 0:
            raise ValueError("Control limits must be positive")
        if self.dt <= 0 or self.min_speed < 0 or self.max_speed <= self.min_speed:
            raise ValueError("Invalid time step or speed limits")


@dataclass
class UAVState:
    """Planar UAV state in metres, metres/step, and radians."""

    x: float
    y: float
    speed: float
    heading: float

    @property
    def position(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float32)

    def copy(self) -> "UAVState":
        return UAVState(self.x, self.y, self.speed, self.heading)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def clip_normalized_parameter(parameter: float | np.ndarray) -> float:
    """Convert the action parameter to a scalar and clip it to ``[-1, 1]``."""

    values = np.asarray(parameter, dtype=np.float32).reshape(-1)
    if values.size != 1:
        raise ValueError("Each benchmark action has exactly one parameter slot")
    return float(np.clip(values[0], -1.0, 1.0))


def advance(
    state: UAVState,
    discrete_action: int,
    normalized_parameter: float | np.ndarray,
    config: DynamicsConfig,
) -> UAVState:
    """Apply one paper-style dynamics step.

    ``MOVE`` changes speed, ``TURN`` changes heading, and ``CATCH`` changes
    neither.  Every action then advances the UAV using its resulting speed and
    heading, matching Eq. (5) and the paper's statement that an unsuccessful
    CATCH advances one step.
    """

    parameter = clip_normalized_parameter(normalized_parameter)
    speed = float(state.speed)
    heading = float(state.heading)

    if discrete_action == MOVE:
        speed += parameter * config.max_acceleration * config.dt
    elif discrete_action == TURN:
        heading += parameter * config.max_turn_angle
    elif discrete_action != CATCH:
        raise ValueError(f"Unknown discrete action: {discrete_action}")

    speed = float(np.clip(speed, config.min_speed, config.max_speed))
    heading = wrap_angle(heading)
    x = state.x + speed * math.cos(heading) * config.dt
    y = state.y + speed * math.sin(heading) * config.dt
    return UAVState(float(x), float(y), speed, heading)


def distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Euclidean distance supporting both scalar and batched goal arrays."""

    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)
    return np.linalg.norm(first_array - second_array, axis=-1)

# ==============================================================================
# Merged from direct_navigation.py
# ==============================================================================

class DirectNavigationEnv(gym.Env):
    """Sparse-reward navigation in a continuous two-dimensional square.

    Actions are ``(discrete_action, parameter)`` where action 0 is MOVE with
    normalized acceleration and action 1 is TURN with normalized angle.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_size: float = 2000.0,
        goal_radius: float = 100.0,
        max_episode_steps: int = 100,
        min_start_goal_distance: float = 0.0,
        boundary_mode: str = "terminate",
        dynamics: DynamicsConfig | None = None,
    ) -> None:
        super().__init__()
        if map_size <= 0 or goal_radius <= 0 or goal_radius >= map_size / 2:
            raise ValueError("Invalid map size or goal radius")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.map_size = float(map_size)
        self.goal_radius = float(goal_radius)
        self.max_episode_steps = int(max_episode_steps)
        self.min_start_goal_distance = float(min_start_goal_distance)
        if boundary_mode not in {"clip", "terminate"}:
            raise ValueError("boundary_mode must be 'clip' or 'terminate'")
        self.boundary_mode = boundary_mode
        self.dynamics = dynamics or DynamicsConfig()

        self.action_space = spaces.Tuple(
            (spaces.Discrete(2), spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32))
        )
        observation_low = np.asarray(
            [0.0, 0.0, self.dynamics.min_speed, -math.pi, 0.0, 0.0],
            dtype=np.float32,
        )
        observation_high = np.asarray(
            [
                self.map_size,
                self.map_size,
                self.dynamics.max_speed,
                math.pi,
                math.sqrt(2.0) * self.map_size,
                float(self.max_episode_steps),
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(observation_low, observation_high, dtype=np.float32),
                "achieved_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
                "desired_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
            }
        )
        self.state = UAVState(0.0, 0.0, 0.0, 0.0)
        self.goal = np.zeros(2, dtype=np.float32)
        self.elapsed_steps = 0

    @property
    def current_phase(self) -> int:
        return 0

    @property
    def num_phases(self) -> int:
        return 1

    @property
    def current_goal(self) -> np.ndarray:
        return self.goal.copy()

    def _sample_start_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(10_000):
            start = self.np_random.uniform(0.0, self.map_size, size=2).astype(np.float32)
            goal = self.np_random.uniform(0.0, self.map_size, size=2).astype(np.float32)
            if float(distance(start, goal)) >= self.min_start_goal_distance:
                return start, goal
        raise RuntimeError("Could not sample start and goal with requested separation")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        sampled_start, sampled_goal = self._sample_start_and_goal()
        start = np.asarray(options.get("start", sampled_start), dtype=np.float32)
        goal = np.asarray(options.get("goal", sampled_goal), dtype=np.float32)
        if start.shape != (2,) or goal.shape != (2,):
            raise ValueError("start and goal must be two-dimensional coordinates")
        if np.any(start < 0) or np.any(start > self.map_size):
            raise ValueError("start must lie inside the map")
        if np.any(goal < 0) or np.any(goal > self.map_size):
            raise ValueError("goal must lie inside the map")
        heading = float(options.get("heading", self.np_random.uniform(-math.pi, math.pi)))
        speed = float(options.get("speed", 0.0))
        self.state = UAVState(float(start[0]), float(start[1]), speed, heading)
        self.goal = goal.copy()
        self.elapsed_steps = 0
        observation = self._get_obs()
        return observation, self._get_info(is_success=self._at_goal(), out_of_bounds=False)

    def _get_obs(self) -> dict[str, np.ndarray]:
        achieved_goal = self.state.position
        vector = np.asarray(
            [
                self.state.x,
                self.state.y,
                self.state.speed,
                self.state.heading,
                float(distance(achieved_goal, self.goal)),
                float(self.elapsed_steps),
            ],
            dtype=np.float32,
        )
        return {
            "observation": vector,
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _at_goal(self) -> bool:
        return bool(distance(self.state.position, self.goal) <= self.goal_radius)

    def _out_of_bounds(self) -> bool:
        position = self.state.position
        return bool(np.any(position < 0.0) or np.any(position > self.map_size))

    def _handle_boundary(self) -> bool:
        """Apply configured boundary behavior and return whether a hit occurred."""
        boundary_hit = self._out_of_bounds()
        if boundary_hit and self.boundary_mode == "clip":
            self.state.x = float(np.clip(self.state.x, 0.0, self.map_size))
            self.state.y = float(np.clip(self.state.y, 0.0, self.map_size))
        return boundary_hit

    def _get_info(self, *, is_success: bool, out_of_bounds: bool) -> dict[str, Any]:
        return {
            "is_success": bool(is_success),
            "out_of_bounds": bool(out_of_bounds),
            "phase": self.current_phase,
            "goal_radius": self.goal_radius,
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> float | np.ndarray:
        del info
        rewards = np.where(distance(achieved_goal, desired_goal) <= self.goal_radius, 0.0, -1.0)
        return float(rewards) if np.ndim(rewards) == 0 else rewards.astype(np.float32)

    def step(
        self, action: tuple[int, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            # Parameters outside [-1, 1] are deliberately accepted and clipped;
            # malformed discrete actions or shapes are still rejected by dynamics.
            discrete_action, parameter = action
            if not self.action_space.spaces[0].contains(discrete_action):
                raise ValueError(f"Invalid discrete action: {discrete_action}")
        else:
            discrete_action, parameter = action
        self.state = advance(self.state, int(discrete_action), parameter, self.dynamics)
        self.elapsed_steps += 1
        boundary_hit = self._handle_boundary()
        is_success = self._at_goal()
        out_of_bounds = boundary_hit and self.boundary_mode == "terminate"
        terminated = bool(is_success or out_of_bounds)
        truncated = bool(self.elapsed_steps >= self.max_episode_steps and not terminated)
        observation = self._get_obs()
        reward = float(self.compute_reward(observation["achieved_goal"], self.goal, None))
        info = self._get_info(is_success=is_success, out_of_bounds=out_of_bounds)
        info["boundary_hit"] = boundary_hit
        return observation, reward, terminated, truncated, info

# ==============================================================================
# Merged from relay_navigation.py
# ==============================================================================

class RelayNavigationEnv(DirectNavigationEnv):
    """Two-stage sparse task: reach/catch a supply, then deliver it.

    The paper requires an explicit CATCH action. The strict reproduction keeps
    that behavior as the default. ``require_catch_action=False`` is retained
    only as a non-paper ablation.
    """

    def __init__(
        self,
        map_size: float = 2000.0,
        goal_radius: float = 100.0,
        relay_radius: float = 100.0,
        max_episode_steps: int = 100,
        min_start_goal_distance: float = 0.0,
        boundary_mode: str = "terminate",
        dynamics: DynamicsConfig | None = None,
        require_catch_action: bool = True,
    ) -> None:
        super().__init__(
            map_size=map_size,
            goal_radius=goal_radius,
            max_episode_steps=max_episode_steps,
            min_start_goal_distance=min_start_goal_distance,
            boundary_mode=boundary_mode,
            dynamics=dynamics,
        )
        if relay_radius <= 0:
            raise ValueError("relay_radius must be positive")
        self.relay_radius = float(relay_radius)
        self.require_catch_action = bool(require_catch_action)
        self.action_space = spaces.Tuple(
            (spaces.Discrete(3), spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32))
        )
        low = np.asarray(
            [0.0, 0.0, self.dynamics.min_speed, -math.pi, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.asarray(
            [
                self.map_size,
                self.map_size,
                self.dynamics.max_speed,
                math.pi,
                math.sqrt(2.0) * self.map_size,
                math.sqrt(2.0) * self.map_size,
                float(self.max_episode_steps),
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low, high, dtype=np.float32),
                "achieved_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
                "desired_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
            }
        )
        self.relay_goal = np.zeros(2, dtype=np.float32)
        self.final_goal = np.zeros(2, dtype=np.float32)
        self.supply_position = np.zeros(2, dtype=np.float32)
        self.phase = 0

    @property
    def current_phase(self) -> int:
        return self.phase

    @property
    def num_phases(self) -> int:
        return 2

    @property
    def current_goal(self) -> np.ndarray:
        return (self.relay_goal if self.phase == 0 else self.final_goal).copy()

    def _sample_three_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(10_000):
            points = self.np_random.uniform(0.0, self.map_size, size=(3, 2)).astype(np.float32)
            if (
                distance(points[0], points[1]) >= self.min_start_goal_distance
                and distance(points[1], points[2]) >= self.min_start_goal_distance
            ):
                return points[0], points[1], points[2]
        raise RuntimeError("Could not sample separated start, relay, and final goals")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # Seed Gymnasium without generating and discarding a direct-navigation task.
        gym.Env.reset(self, seed=seed)
        options = options or {}
        sampled_start, sampled_relay, sampled_final = self._sample_three_points()
        start = np.asarray(options.get("start", sampled_start), dtype=np.float32)
        relay = np.asarray(options.get("relay_goal", sampled_relay), dtype=np.float32)
        final = np.asarray(options.get("final_goal", sampled_final), dtype=np.float32)
        for name, point in (("start", start), ("relay_goal", relay), ("final_goal", final)):
            if point.shape != (2,) or np.any(point < 0) or np.any(point > self.map_size):
                raise ValueError(f"{name} must be a two-dimensional point inside the map")
        self.relay_goal = relay.copy()
        self.supply_position = relay.copy()
        self.final_goal = final.copy()
        self.goal = self.final_goal  # Compatibility with the direct environment internals.
        self.phase = 0
        self.elapsed_steps = 0
        heading = float(options.get("heading", self.np_random.uniform(-math.pi, math.pi)))
        speed = float(options.get("speed", 0.0))
        self.state = UAVState(float(start[0]), float(start[1]), speed, heading)
        observation = self._get_obs()
        return observation, self._get_info(is_success=False, out_of_bounds=False)

    def _at_relay(self) -> bool:
        return bool(distance(self.state.position, self.relay_goal) <= self.relay_radius)

    def _at_goal(self) -> bool:
        return bool(self.phase == 1 and distance(self.supply_position, self.final_goal) <= self.goal_radius)

    def _get_obs(self) -> dict[str, np.ndarray]:
        # Paper GSM semantics: before pickup, achieved_goal is the UAV position;
        # after pickup, achieved_goal is the carried supply position.
        achieved_goal = self.state.position if self.phase == 0 else self.supply_position.copy()
        active_goal = self.current_goal
        vector = np.asarray(
            [
                self.state.x,
                self.state.y,
                self.state.speed,
                self.state.heading,
                float(distance(achieved_goal, self.final_goal)),
                float(distance(achieved_goal, self.relay_goal)),
                float(self.elapsed_steps),
                float(self.phase),
            ],
            dtype=np.float32,
        )
        return {
            "observation": vector,
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": active_goal,
        }

    def _get_info(self, *, is_success: bool, out_of_bounds: bool) -> dict[str, Any]:
        return {
            "is_success": bool(is_success),
            "out_of_bounds": bool(out_of_bounds),
            "phase": self.phase,
            "num_phases": self.num_phases,
            "relay_reached": self._at_relay() if self.phase == 0 else True,
            "carrying_supply": self.phase == 1,
            "goal_radius": self.goal_radius,
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> float | np.ndarray:
        # The paper defines HER through dynamically switched achieved/desired
        # goals. It does not specify an additional action-consistency test in
        # the relabelled reward, so strict reproduction uses the same sparse
        # positional goal test as ordinary HER.
        return super().compute_reward(achieved_goal, desired_goal, info)

    def step(
        self, action: tuple[int, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        discrete_action, parameter = action
        if not self.action_space.spaces[0].contains(discrete_action):
            raise ValueError(f"Invalid discrete action: {discrete_action}")

        was_phase = self.phase
        at_relay_before_action = self._at_relay()
        if self.phase == 0 and self.require_catch_action and discrete_action == CATCH and at_relay_before_action:
            self.phase = 1

        self.state = advance(self.state, int(discrete_action), parameter, self.dynamics)
        if self.phase == 1:
            self.supply_position = self.state.position.copy()
        self.elapsed_steps += 1
        boundary_hit = self._handle_boundary()
        if self.phase == 0 and not self.require_catch_action and self._at_relay():
            self.phase = 1

        is_success = self._at_goal()
        out_of_bounds = boundary_hit and self.boundary_mode == "terminate"
        terminated = bool(is_success or out_of_bounds)
        truncated = bool(self.elapsed_steps >= self.max_episode_steps and not terminated)
        observation = self._get_obs()
        # The paper gives no intermediate success reward: only final delivery is 0.
        reward = 0.0 if is_success else -1.0
        info = self._get_info(is_success=is_success, out_of_bounds=out_of_bounds)
        info["boundary_hit"] = boundary_hit
        info["phase_changed"] = was_phase != self.phase
        return observation, reward, terminated, truncated, info

# ==============================================================================
# Merged from multi_relay_navigation.py
# ==============================================================================

class MultiRelayNavigationEnv(DirectNavigationEnv):
    """Navigate through N ordered relay points and then the final goal.

    Each relay requires a valid CATCH by default. Rewards remain -1 until the
    final goal is reached; increasing N never changes reward density.
    """

    def __init__(
        self,
        num_relays: int = 1,
        map_size: float = 2000.0,
        goal_radius: float = 100.0,
        relay_radius: float = 100.0,
        max_episode_steps: int = 100,
        min_start_goal_distance: float = 0.0,
        boundary_mode: str = "terminate",
        dynamics: DynamicsConfig | None = None,
        require_catch_action: bool = True,
    ) -> None:
        if int(num_relays) != num_relays or num_relays < 0:
            raise ValueError("num_relays must be a non-negative integer")
        super().__init__(
            map_size=map_size,
            goal_radius=goal_radius,
            max_episode_steps=max_episode_steps,
            min_start_goal_distance=min_start_goal_distance,
            boundary_mode=boundary_mode,
            dynamics=dynamics,
        )
        if relay_radius <= 0:
            raise ValueError("relay_radius must be positive")
        self.num_relays = int(num_relays)
        self.relay_radius = float(relay_radius)
        self.require_catch_action = bool(require_catch_action)
        self.relay_goals = np.zeros((self.num_relays, 2), dtype=np.float32)
        self.final_goal = np.zeros(2, dtype=np.float32)
        self.phase = 0
        num_actions = 3 if self.num_relays else 2
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),
            spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        ))
        # [x,y,v,theta,d_final,d_relay_1..d_relay_N,steps,phase]
        low = np.asarray(
            [0.0, 0.0, self.dynamics.min_speed, -math.pi, 0.0]
            + [0.0] * self.num_relays + [0.0, 0.0], dtype=np.float32)
        high = np.asarray(
            [self.map_size, self.map_size, self.dynamics.max_speed, math.pi,
             math.sqrt(2.0) * self.map_size]
            + [math.sqrt(2.0) * self.map_size] * self.num_relays
            + [float(self.max_episode_steps), float(max(1, self.num_relays))], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low, high, dtype=np.float32),
            "achieved_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
            "desired_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
        })

    @property
    def current_phase(self) -> int:
        return self.phase

    @property
    def num_phases(self) -> int:
        return self.num_relays + 1

    @property
    def current_goal(self) -> np.ndarray:
        if self.phase < self.num_relays:
            return self.relay_goals[self.phase].copy()
        return self.final_goal.copy()

    def _sample_route(self) -> np.ndarray:
        count = self.num_relays + 2  # start + relays + final
        for _ in range(10_000):
            points = self.np_random.uniform(
                0.0, self.map_size, size=(count, 2)).astype(np.float32)
            segment_distances = distance(points[:-1], points[1:])
            if np.all(segment_distances >= self.min_start_goal_distance):
                return points
        raise RuntimeError("Could not sample a route with requested segment separation")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        options = options or {}
        sampled = self._sample_route()
        start = np.asarray(options.get("start", sampled[0]), dtype=np.float32)
        relays = np.asarray(options.get("relay_goals", sampled[1:-1]), dtype=np.float32)
        if self.num_relays == 0:
            relays = relays.reshape(0, 2)
        final = np.asarray(options.get("final_goal", sampled[-1]), dtype=np.float32)
        if start.shape != (2,) or final.shape != (2,) or relays.shape != (self.num_relays, 2):
            raise ValueError("start/final must be [2] and relay_goals must be [num_relays, 2]")
        for name, points in (("start", start[None]), ("relay_goals", relays), ("final_goal", final[None])):
            if np.any(points < 0) or np.any(points > self.map_size):
                raise ValueError(f"{name} must lie inside the map")
        self.relay_goals = relays.copy()
        self.final_goal = final.copy()
        self.goal = self.final_goal
        self.phase = 0
        self.elapsed_steps = 0
        self.state = UAVState(
            float(start[0]), float(start[1]),
            float(options.get("speed", 0.0)),
            float(options.get("heading", self.np_random.uniform(-math.pi, math.pi))),
        )
        obs = self._get_obs()
        return obs, self._get_info(is_success=False, out_of_bounds=False)

    def _at_current_relay(self) -> bool:
        return bool(
            self.phase < self.num_relays
            and distance(self.state.position, self.relay_goals[self.phase]) <= self.relay_radius)

    def _at_goal(self) -> bool:
        return bool(
            self.phase == self.num_relays
            and distance(self.state.position, self.final_goal) <= self.goal_radius)

    def _get_obs(self) -> dict[str, np.ndarray]:
        achieved = self.state.position
        relay_distances = [float(distance(achieved, goal)) for goal in self.relay_goals]
        vector = np.asarray([
            self.state.x, self.state.y, self.state.speed, self.state.heading,
            float(distance(achieved, self.final_goal)),
            *relay_distances,
            float(self.elapsed_steps), float(self.phase),
        ], dtype=np.float32)
        return {
            "observation": vector,
            "achieved_goal": achieved.copy(),
            "desired_goal": self.current_goal,
        }

    def _get_info(self, *, is_success: bool, out_of_bounds: bool) -> dict[str, Any]:
        return {
            "is_success": bool(is_success),
            "out_of_bounds": bool(out_of_bounds),
            "phase": self.phase,
            "num_phases": self.num_phases,
            "relays_completed": min(self.phase, self.num_relays),
            "all_relays_completed": self.phase == self.num_relays,
            "goal_radius": self.goal_radius,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Use CATCH-consistent virtual success for unfinished relay stages."""
        if not isinstance(info, dict) or not info.get("is_her", False):
            return super().compute_reward(achieved_goal, desired_goal, info)
        source_phase = int(info.get("her_source_phase", info.get("phase", 0)))
        if source_phase >= self.num_relays or not self.require_catch_action:
            return super().compute_reward(achieved_goal, desired_goal, info)
        before = np.asarray(
            info.get("her_achieved_goal_before", achieved_goal), dtype=np.float32)
        success = (
            int(info.get("her_source_action", -1)) == CATCH
            and distance(before, desired_goal) <= self.relay_radius
        )
        rewards = np.where(success, 0.0, -1.0)
        return float(rewards) if np.ndim(rewards) == 0 else rewards.astype(np.float32)

    def step(self, action):
        discrete_action, parameter = action
        if not self.action_space.spaces[0].contains(discrete_action):
            raise ValueError(f"Invalid discrete action: {discrete_action}")
        previous_phase = self.phase
        if (
            self.require_catch_action and discrete_action == CATCH
            and self._at_current_relay()
        ):
            self.phase += 1
        self.state = advance(self.state, int(discrete_action), parameter, self.dynamics)
        self.elapsed_steps += 1
        boundary_hit = self._handle_boundary()
        if not self.require_catch_action and self._at_current_relay():
            self.phase += 1
        success = self._at_goal()
        out_of_bounds = boundary_hit and self.boundary_mode == "terminate"
        terminated = bool(success or out_of_bounds)
        truncated = bool(self.elapsed_steps >= self.max_episode_steps and not terminated)
        obs = self._get_obs()
        info = self._get_info(is_success=success, out_of_bounds=out_of_bounds)
        info["boundary_hit"] = boundary_hit
        info["phase_changed"] = self.phase != previous_phase
        return obs, 0.0 if success else -1.0, terminated, truncated, info

# ==============================================================================
# Merged from wrappers.py
# ==============================================================================

class GoalObservationEncoder:
    """Normalize the paper state, optionally followed by the HER goal.

    The paper's P-DQN/MP-DQN baselines consume only ``s``.  Goal-conditioned
    HER variants consume ``(s, g)``.  Relay environments expose an explicit
    phase for CT-WM trajectory analysis, but the paper baseline state does not
    include it, so callers can exclude that coordinate without removing it
    from the environment API.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        include_goal: bool = True,
        excluded_state_indices: tuple[int, ...] = (),
    ) -> None:
        state_space = observation_space["observation"]
        goal_space = observation_space["desired_goal"]
        if not isinstance(state_space, spaces.Box) or not isinstance(goal_space, spaces.Box):
            raise TypeError("GoalObservationEncoder requires Box state and goal spaces")
        state_low = state_space.low.reshape(-1)
        state_high = state_space.high.reshape(-1)
        excluded = {index % state_low.size for index in excluded_state_indices}
        self.state_indices = np.asarray(
            [index for index in range(state_low.size) if index not in excluded], dtype=np.int64)
        if not self.state_indices.size:
            raise ValueError("Cannot exclude every state coordinate")
        self.include_goal = bool(include_goal)
        lows = [state_low[self.state_indices]]
        highs = [state_high[self.state_indices]]
        if self.include_goal:
            lows.append(goal_space.low.reshape(-1))
            highs.append(goal_space.high.reshape(-1))
        self.low = np.concatenate(lows).astype(np.float32)
        self.high = np.concatenate(highs).astype(np.float32)
        if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
            raise ValueError("Observation normalization bounds must be finite")
        if np.any(self.high <= self.low):
            raise ValueError("Every observation bound must have positive range")

    @property
    def output_dim(self) -> int:
        return int(self.low.size)

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
        parts = [state[self.state_indices]]
        if self.include_goal:
            parts.append(np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1))
        raw = np.concatenate(parts)
        if raw.shape != self.low.shape:
            raise ValueError(f"Expected flattened observation {self.low.shape}, got {raw.shape}")
        normalized = 2.0 * (raw - self.low) / (self.high - self.low) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)


__all__ = [
    "DynamicsConfig", "UAVState", "MOVE", "TURN", "CATCH",
    "advance", "distance", "wrap_angle", "clip_normalized_parameter",
    "DirectNavigationEnv", "RelayNavigationEnv", "MultiRelayNavigationEnv",
    "GoalObservationEncoder",
]
