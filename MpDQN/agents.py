"""P-DQN and MP-DQN agents for parameterized UAV actions.

Merged from common.py, networks.py, pdqn.py, mpdqn.py, and package __init__.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import copy
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ParameterizedActionSpec:
    """Map each discrete action to its continuous parameter-vector slice."""
    parameter_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.parameter_sizes)
        if not sizes or any(size < 0 for size in sizes):
            raise ValueError("parameter_sizes must be a non-empty sequence of non-negative integers")
        if sum(sizes) <= 0:
            raise ValueError("At least one action must have a continuous parameter")
        object.__setattr__(self, "parameter_sizes", sizes)

    @property
    def num_actions(self) -> int:
        return len(self.parameter_sizes)

    @property
    def total_parameter_dim(self) -> int:
        return sum(self.parameter_sizes)

    def parameter_slice(self, action: int) -> slice:
        if action < 0 or action >= self.num_actions:
            raise IndexError(f"Discrete action {action} is out of range")
        start = sum(self.parameter_sizes[:action])
        return slice(start, start + self.parameter_sizes[action])

    def selected_parameter(self, parameters: np.ndarray, action: int) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float32)
        if values.shape != (self.total_parameter_dim,):
            raise ValueError(f"Expected parameter vector {(self.total_parameter_dim,)}, got {values.shape}")
        selected = values[self.parameter_slice(action)]
        return selected.copy() if selected.size else np.zeros(1, dtype=np.float32)

    def masks(self, *, device: torch.device | str | None = None) -> Tensor:
        masks = torch.zeros(self.num_actions, self.total_parameter_dim, dtype=torch.float32, device=device)
        for action in range(self.num_actions):
            masks[action, self.parameter_slice(action)] = 1.0
        return masks


@dataclass(frozen=True)
class ActionSelection:
    discrete_action: int
    selected_parameter: np.ndarray
    all_parameters: np.ndarray
    q_values: np.ndarray

    def environment_action(self) -> tuple[int, np.ndarray]:
        return self.discrete_action, self.selected_parameter.copy()


@dataclass
class TransitionBatch:
    states: Tensor
    actions: Tensor
    action_parameters: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor

    def __post_init__(self) -> None:
        batch_size = self.states.shape[0]
        if self.states.ndim != 2 or self.next_states.shape != self.states.shape:
            raise ValueError("states and next_states must have shape [batch, state_dim]")
        if self.action_parameters.ndim != 2 or self.action_parameters.shape[0] != batch_size:
            raise ValueError("action_parameters must have shape [batch, parameter_dim]")
        for name in ("actions", "rewards", "dones"):
            if getattr(self, name).numel() != batch_size:
                raise ValueError(f"{name} must contain one value per transition")

    @classmethod
    def from_numpy(cls, *, states, actions, action_parameters, rewards, next_states, dones, device="cpu"):
        return cls(
            states=torch.as_tensor(states, dtype=torch.float32, device=device),
            actions=torch.as_tensor(actions, dtype=torch.long, device=device).reshape(-1),
            action_parameters=torch.as_tensor(action_parameters, dtype=torch.float32, device=device),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1),
            next_states=torch.as_tensor(next_states, dtype=torch.float32, device=device),
            dones=torch.as_tensor(dones, dtype=torch.float32, device=device).reshape(-1),
        )

    def to(self, device):
        return TransitionBatch(**{name: getattr(self, name).to(device) for name in (
            "states", "actions", "action_parameters", "rewards", "next_states", "dones")})


def goal_observation_to_vector(observation: Mapping[str, Any]) -> np.ndarray:
    state = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
    goal = np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1)
    return np.concatenate([state, goal], dtype=np.float32)


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 < tau <= 1.0: raise ValueError("tau must be in (0, 1]")
    for tp, sp in zip(target.parameters(), source.parameters()): tp.lerp_(sp, tau)


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary); temporary.replace(path)


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0: raise ValueError("Network input and output dimensions must be positive")
    layers = []; previous = input_dim
    for size in hidden_sizes:
        if size <= 0: raise ValueError("Hidden layer sizes must be positive")
        linear = nn.Linear(previous, int(size)); nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu"); nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ReLU())); previous = int(size)
    output = nn.Linear(previous, output_dim); nn.init.uniform_(output.weight, -3e-3, 3e-3); nn.init.zeros_(output.bias); layers.append(output)
    return nn.Sequential(*layers)


class ParameterActor(nn.Module):
    def __init__(self, state_dim, parameter_dim, hidden_sizes=(128, 64)):
        super().__init__(); self.network = build_mlp(state_dim, hidden_sizes, parameter_dim)
    def forward(self, states): return torch.tanh(self.network(states))


class QNetwork(nn.Module):
    def __init__(self, state_dim, parameter_dim, num_actions, hidden_sizes=(128, 64)):
        super().__init__(); self.network = build_mlp(state_dim + parameter_dim, hidden_sizes, num_actions)
    def forward(self, states, action_parameters):
        if states.ndim != 2 or action_parameters.ndim != 2: raise ValueError("Q-network inputs must be rank-two batches")
        return self.network(torch.cat((states, action_parameters), dim=-1))


def build_multipass_parameters(parameters: Tensor, spec: ParameterizedActionSpec) -> Tensor:
    if parameters.ndim != 2 or parameters.shape[-1] != spec.total_parameter_dim:
        raise ValueError(f"Expected parameters [batch, {spec.total_parameter_dim}], got {tuple(parameters.shape)}")
    masks = spec.masks(device=parameters.device).to(parameters.dtype)
    return parameters[:, None, :] * masks[None, :, :]


def multipass_q_values(q_network, states, parameters, spec):
    masked = build_multipass_parameters(parameters, spec); batch_size = states.shape[0]
    repeated_states = states[:, None, :].expand(-1, spec.num_actions, -1)
    all_outputs = q_network(repeated_states.reshape(batch_size * spec.num_actions, -1), masked.reshape(batch_size * spec.num_actions, -1)).reshape(batch_size, spec.num_actions, spec.num_actions)
    return all_outputs.diagonal(dim1=1, dim2=2)


@dataclass(frozen=True)
class PDQNConfig:
    state_dim: int
    parameter_sizes: tuple[int, ...]
    hidden_sizes: tuple[int, ...] = (128, 64)
    gamma: float = 0.99
    tau: float = 0.005
    q_learning_rate: float = 1e-3
    parameter_learning_rate: float = 1e-4
    gradient_clip_norm: float = 10.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        if self.state_dim <= 0: raise ValueError("state_dim must be positive")
        ParameterizedActionSpec(self.parameter_sizes)
        if not 0 <= self.gamma <= 1: raise ValueError("gamma must be in [0, 1]")
        if not 0 < self.tau <= 1: raise ValueError("tau must be in (0, 1]")
        if self.q_learning_rate <= 0 or self.parameter_learning_rate <= 0: raise ValueError("Learning rates must be positive")


class PDQNAgent:
    algorithm_name = "pdqn"
    def __init__(self, config):
        self.config=config; self.spec=ParameterizedActionSpec(config.parameter_sizes); self.device=torch.device(config.device); self.rng=seed_everything(config.seed)
        self.parameter_actor=ParameterActor(config.state_dim,self.spec.total_parameter_dim,config.hidden_sizes).to(self.device)
        self.q_network=QNetwork(config.state_dim,self.spec.total_parameter_dim,self.spec.num_actions,config.hidden_sizes).to(self.device)
        self.target_parameter_actor=copy.deepcopy(self.parameter_actor).to(self.device).eval(); self.target_q_network=copy.deepcopy(self.q_network).to(self.device).eval()
        for network in (self.target_parameter_actor,self.target_q_network):
            for parameter in network.parameters(): parameter.requires_grad_(False)
        self.q_optimizer=torch.optim.Adam(self.q_network.parameters(),lr=config.q_learning_rate); self.parameter_optimizer=torch.optim.Adam(self.parameter_actor.parameters(),lr=config.parameter_learning_rate); self.update_steps=0
    def _q_values(self,network,states,parameters): return network(states,parameters)
    @torch.no_grad()
    def q_values(self,states,parameters):
        s=torch.as_tensor(states,dtype=torch.float32,device=self.device); p=torch.as_tensor(parameters,dtype=torch.float32,device=self.device)
        if s.ndim==1:s=s.unsqueeze(0)
        if p.ndim==1:p=p.unsqueeze(0)
        return self._q_values(self.q_network,s,p).cpu().numpy()
    @torch.no_grad()
    def select_action(self,state,*,epsilon=0.0,parameter_noise_std=0.0):
        if not 0<=epsilon<=1 or parameter_noise_std<0: raise ValueError("Invalid exploration settings")
        state=np.asarray(state,dtype=np.float32)
        if state.shape!=(self.config.state_dim,): raise ValueError(f"Expected state shape {(self.config.state_dim,)}, got {state.shape}")
        parameters=self.parameter_actor(torch.as_tensor(state,device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        if parameter_noise_std:
            parameters+=self.rng.normal(0,parameter_noise_std,size=parameters.shape); parameters=np.clip(parameters,-1,1).astype(np.float32)
        q_values=self.q_values(state,parameters)[0]; discrete_action=int(self.rng.integers(self.spec.num_actions)) if self.rng.random()<epsilon else int(np.argmax(q_values))
        return ActionSelection(discrete_action,self.spec.selected_parameter(parameters,discrete_action),parameters.copy(),q_values.copy())
    def update(self,batch):
        batch=batch.to(self.device)
        with torch.no_grad():
            np_=self.target_parameter_actor(batch.next_states); nq=self._q_values(self.target_q_network,batch.next_states,np_).max(dim=1).values; targets=batch.rewards+self.config.gamma*(1-batch.dones)*nq
        predicted_all=self._q_values(self.q_network,batch.states,batch.action_parameters); predicted=predicted_all.gather(1,batch.actions[:,None]).squeeze(1); q_loss=F.mse_loss(predicted,targets)
        self.q_optimizer.zero_grad(set_to_none=True); q_loss.backward(); qgn=torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),self.config.gradient_clip_norm); self.q_optimizer.step()
        for p in self.q_network.parameters():p.requires_grad_(False)
        ap=self.parameter_actor(batch.states); aq=self._q_values(self.q_network,batch.states,ap); aloss=-aq.sum(dim=1).mean(); self.parameter_optimizer.zero_grad(set_to_none=True); aloss.backward(); pgn=torch.nn.utils.clip_grad_norm_(self.parameter_actor.parameters(),self.config.gradient_clip_norm); self.parameter_optimizer.step()
        for p in self.q_network.parameters():p.requires_grad_(True)
        soft_update(self.target_q_network,self.q_network,self.config.tau); soft_update(self.target_parameter_actor,self.parameter_actor,self.config.tau); self.update_steps+=1
        return {"q_loss":float(q_loss.detach().cpu()),"parameter_actor_loss":float(aloss.detach().cpu()),"q_gradient_norm":float(torch.as_tensor(qgn).cpu()),"parameter_gradient_norm":float(torch.as_tensor(pgn).cpu()),"mean_q":float(predicted.detach().mean().cpu()),"mean_target_q":float(targets.detach().mean().cpu()),"update_steps":float(self.update_steps)}
    def checkpoint(self):
        return {"algorithm":self.algorithm_name,"config":asdict(self.config),"parameter_actor":self.parameter_actor.state_dict(),"q_network":self.q_network.state_dict(),"target_parameter_actor":self.target_parameter_actor.state_dict(),"target_q_network":self.target_q_network.state_dict(),"parameter_optimizer":self.parameter_optimizer.state_dict(),"q_optimizer":self.q_optimizer.state_dict(),"update_steps":self.update_steps,"exploration_rng_state":self.rng.bit_generator.state}
    def save(self,path): save_checkpoint(path,self.checkpoint())
    def load(self,path):
        payload=torch.load(path,map_location=self.device,weights_only=False)
        if payload.get("algorithm")!=self.algorithm_name: raise ValueError(f"Checkpoint algorithm {payload.get('algorithm')!r} does not match {self.algorithm_name!r}")
        for name in ("parameter_actor","q_network","target_parameter_actor","target_q_network"): getattr(self,name).load_state_dict(payload[name])
        self.parameter_optimizer.load_state_dict(payload["parameter_optimizer"]); self.q_optimizer.load_state_dict(payload["q_optimizer"]); self.update_steps=int(payload["update_steps"])
        if "exploration_rng_state" in payload:self.rng.bit_generator.state=dict(payload["exploration_rng_state"])


class MPDQNAgent(PDQNAgent):
    algorithm_name = "mpdqn"
    def _q_values(self, network: nn.Module, states: Tensor, parameters: Tensor) -> Tensor:
        return multipass_q_values(network, states, parameters, self.spec)


__all__ = ["ActionSelection", "MPDQNAgent", "PDQNAgent", "PDQNConfig", "ParameterizedActionSpec", "TransitionBatch", "goal_observation_to_vector", "save_checkpoint"]
