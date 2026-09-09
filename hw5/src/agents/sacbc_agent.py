from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class SACBCAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_beta,
        make_beta_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.beta = make_beta()

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.beta_optimizer = make_beta_optimizer(self.beta.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha

        self.target_entropy = -action_dim / 2  # Heuristic value (|A| / 2) from the SAC paper.

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # Get the mode action from a tanh transformed distribution.
        action = self.actor(observation).base_dist.base_dist.mode.tanh()
        return ptu.to_numpy(action[0])

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # HƯỚNG DẪN (soft Bellman backup như SAC gốc, nhưng offline nên actor được
        # đánh giá tại next_observations thay vì rollout thật):
        #   1. Với torch.no_grad():
        #        a. Lấy phân phối actor tại s': next_dist = self.actor(next_observations)
        #        b. Lấy hành động mẫu (reparameterized): next_actions = next_dist.rsample()
        #        c. next_log_probs = next_dist.log_prob(next_actions)
        #        d. next_q = self.target_critic(next_observations, next_actions).min(dim=0).values
        #        e. Trừ đi entropy bonus (soft value):
        #             next_v = next_q - self.beta() * next_log_probs
        #        f. target = rewards + self.discount * (1 - dones) * next_v
        #   2. q = self.critic(observations, actions)  -> shape (n_ensembles, B), CÓ gradient
        #   3. loss = ((q - target[None, :]) ** 2).mean()
        q = ...
        loss = ...

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor
        """
        # TODO(student): Compute the actor loss
        # HƯỚNG DẪN (SAC+BC = SAC objective + behavior-cloning regularizer):
        # Actor loss gồm 3 phần cộng lại: q_loss + bc_loss + entropy_loss (đã có sẵn
        # dòng "loss = q_loss + bc_loss + entropy_loss" phía dưới).
        #   1. Lấy phân phối & action mẫu (CÓ gradient, dùng rsample để backprop được):
        #        dist = self.actor(observations)
        #        actor_actions = dist.rsample()
        #        log_probs = dist.log_prob(actor_actions)
        #   2. q_loss: muốn actor chọn action tối đa hoá Q -> minimize -Q
        #        q = self.critic(observations, actor_actions).min(dim=0).values
        #        q_loss = -q.mean()
        #   3. bc_loss: phạt nếu action của actor lệch xa action trong dataset (giữ actor
        #      gần behavior policy, tránh out-of-distribution actions):
        #        mses = torch.mean((actor_actions - actions) ** 2, dim=-1)   # (B,)
        #        bc_loss = self.alpha * mses.mean()
        #   4. entropy_loss: phần entropy bonus của SAC (giữ policy có tính khám phá).
        #      Dùng self.beta().detach() để KHÔNG lan gradient vào beta ở bước này
        #      (beta được train riêng trong update_beta()):
        #        entropy_loss = (self.beta().detach() * log_probs).mean()
        q_loss = ...

        mses = ...
        bc_loss = ...

        entropy_loss = ...

        loss = q_loss + bc_loss + entropy_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "total_loss": loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "entropy_loss": entropy_loss,
            "mse": mses.mean(),
        }

    @torch.compile
    def update_beta(
        self,
        observations: torch.Tensor,
    ):
        """
        Update the beta parameter using dual gradient descent.
        """
        actor_dists = self.actor(observations)
        actor_actions = actor_dists.rsample()
        log_probs = actor_dists.log_prob(actor_actions)

        loss = self.beta() * (-log_probs - self.target_entropy).detach().mean()

        self.beta_optimizer.zero_grad()
        loss.backward()
        self.beta_optimizer.step()

        return {
            "beta_loss": loss,
            "beta": self.beta(),
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics_beta = self.update_beta(observations)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"beta/{k}": v.item() for k, v in metrics_beta.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        # HƯỚNG DẪN: giống hệt IQL — soft update:
        #   target_param <- (1 - tau) * target_param + tau * param, tau = self.target_update_rate
        #   with torch.no_grad():
        #       for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
        #           target_param.data.mul_(1 - self.target_update_rate)
        #           target_param.data.add_(self.target_update_rate * param.data)
        ...
