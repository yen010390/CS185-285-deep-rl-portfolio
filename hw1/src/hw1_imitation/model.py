"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor, *, num_steps: int = 10) -> torch.Tensor:
        pass


class MSEPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, chunk_size, hidden_dims=(256, 256, 256)):
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(self, state, action_chunk):
        pred = self.net(state).view(state.shape[0], self.chunk_size, self.action_dim)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state, **kwargs):
        with torch.no_grad():
            return self.net(state).view(state.shape[0], self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, chunk_size, hidden_dims=(256, 256, 256)):
        super().__init__(state_dim, action_dim, chunk_size)
        # Input: state + flattened noisy action + scalar timestep tau
        input_dim = state_dim + (chunk_size * action_dim) + 1
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state, noisy_action, tau):
        # tau: (B, 1)
        x = torch.cat([state, noisy_action.view(state.shape[0], -1), tau], dim=1)
        return self.net(x).view(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(self, state, action_chunk):
        batch_size = state.shape[0]
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(batch_size, 1, device=state.device)

        # Spec eq.2: A_{t,tau} = tau*A_t + (1-tau)*A_{t,0}
        t = tau.view(-1, 1, 1)
        noisy_action = t * action_chunk + (1.0 - t) * noise

        # Target velocity: d/dtau A_{t,tau} = A_t - A_{t,0}
        target_velocity = action_chunk - noise

        v_pred = self.forward(state, noisy_action, tau)
        return nn.functional.mse_loss(v_pred, target_velocity)

    def sample_actions(self, state, *, num_steps=10):
        """Euler integration from tau=0 to tau=1."""
        with torch.no_grad():
            batch_size = state.shape[0]
            x = torch.randn(batch_size, self.chunk_size, self.action_dim, device=state.device)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                tau = torch.full((batch_size, 1), i * dt, device=state.device)
                v = self.forward(state, x, tau)
                x = x + v * dt
            # Do NOT clamp here — actions are z-scored normalized.
            # clamp(±1) only covers 68% of the action distribution.
            # evaluation.py already clips to action_space bounds after
            # denormalization, so no clamping is needed here.
            return x


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone() for name, param in model.named_parameters()
        }
        self._backup: dict[str, torch.Tensor] = {}

    def update(self) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        """Backup current weights, then load EMA weights into model."""
        self._backup = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original weights after evaluation."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._backup:
                    param.data.copy_(self._backup[name])
        self._backup = {}


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (256, 256, 256),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")