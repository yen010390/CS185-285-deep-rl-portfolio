from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # Epsilon-greedy: với xác suất epsilon, chọn hành động ngẫu nhiên (khám phá);
        # ngược lại chọn hành động có Q-value cao nhất theo self.critic (khai thác).
        qa_values = self.critic(observation)  # shape (1, num_actions)

        if np.random.rand() < epsilon:
            action = torch.tensor(
                [np.random.randint(self.num_actions)], device=qa_values.device
            )
        else:
            action = qa_values.argmax(dim=1)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            next_qa_values = self.target_critic(next_obs)  # (batch_size, num_actions)

            if self.use_double_q:
                # Double DQN: chọn hành động bằng online network (self.critic),
                # nhưng đánh giá giá trị của hành động đó bằng target network.
                next_action = self.critic(next_obs).argmax(dim=1)
            else:
                # Vanilla DQN: chọn và đánh giá hành động đều bằng target_critic.
                next_action = next_qa_values.argmax(dim=1)

            next_q_values = torch.gather(
                next_qa_values, dim=1, index=next_action.unsqueeze(1)
            ).squeeze(1)
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            # target = r + gamma * (1 - done) * Q_target(s', a')
            target_values = reward + self.discount * (1 - done.float()) * next_q_values
            assert target_values.shape == (batch_size,), target_values.shape

        # Train the critic with the target values
        qa_values = self.critic(obs)  # (batch_size, num_actions)
        q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)
        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats