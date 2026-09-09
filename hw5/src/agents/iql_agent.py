from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        expectile: float,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.expectile = expectile

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.actor(observation).mode  # Take the mean (mode) action
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action[0])

    @staticmethod
    def iql_expectile_loss(
        adv: torch.Tensor, expectile: float,
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        # TODO(student): Implement the expectile loss
        # HƯỚNG DẪN:
        # Expectile loss (còn gọi L2_tau) dùng để hồi quy V(s) sao cho nó xấp xỉ
        # "expectile thứ tau" của phân phối Q(s,a), thay vì trung bình (mean) như MSE thường.
        #   L2_tau(u) = |tau - 1{u < 0}| * u^2
        # Trong đó `adv` chính là u = Q(s,a) - V(s) (residual/advantage).
        # Cách làm:
        #   1. Tạo trọng số `weight`: nếu adv >= 0 thì weight = expectile (tau),
        #      ngược lại (adv < 0) thì weight = 1 - expectile.
        #      -> Dùng torch.where(adv >= 0, expectile, 1 - expectile)
        #   2. Trả về weight * adv**2 (KHÔNG lấy mean ở đây — hàm này trả về loss
        #      theo từng phần tử, việc .mean() sẽ được gọi ở nơi dùng hàm này).
        return ...

    @torch.compile
    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update V(s) with expectile regression
        """
        # TODO(student): Compute the value loss
        # HƯỚNG DẪN:
        # V(s) được fit để xấp xỉ "expectile" của Q_target(s,a) (dùng target_critic,
        # KHÔNG dùng self.critic, và phải torch.no_grad() vì V không được lan truyền
        # gradient ngược vào critic).
        #   1. Với torch.no_grad(): tính target_q = self.target_critic(observations, actions)
        #      -> shape (n_ensembles, B). Lấy min qua chiều ensemble (dim=0) để có
        #      pessimistic estimate: target_q.min(dim=0).values -> shape (B,)
        #   2. Tính v = self.value(observations)  (CÓ gradient, vì đây là cái ta đang train)
        #   3. adv = target_q - v
        #   4. loss = self.iql_expectile_loss(adv, self.expectile).mean()
        v = ...
        loss = ...

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

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
        # HƯỚNG DẪN:
        # Điểm đặc biệt của IQL: Q không cần max/policy ở s' để bootstrap, mà dùng
        # trực tiếp V(s') (đã được train ở update_v, gọi trước update_q trong update()).
        #   1. Với torch.no_grad():
        #        next_v = self.value(next_observations)          # shape (B,)
        #        target = rewards + self.discount * (1 - dones) * next_v   # shape (B,)
        #   2. q = self.critic(observations, actions)   -> shape (n_ensembles, B), CÓ gradient
        #   3. loss = MSE giữa q và target, nhớ broadcast target lên chiều ensemble:
        #        loss = ((q - target[None, :]) ** 2).mean()
        #      (mỗi ensemble member đều được train khớp cùng 1 target — kiểu "twin Q")
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
        Update the actor using advantage-weighted regression
        """
        # TODO(student): Compute the actor loss
        # HƯỚNG DẪN (Advantage-Weighted Regression - AWR):
        # Ý tưởng: cho actor học behavior cloning trên các action tốt hơn V(s) hiện tại,
        # trọng số theo exp(alpha * advantage) — action càng "tốt" so với baseline V(s)
        # thì trọng số càng lớn.
        #   1. Với torch.no_grad() (advantage KHÔNG lan truyền gradient vào actor):
        #        target_q = self.target_critic(observations, actions).min(dim=0).values
        #        v = self.value(observations)
        #        adv = target_q - v
        #        exp_adv = torch.exp(self.alpha * adv)
        #        # Nên clamp exp_adv (vd .clamp(max=100.0)) để tránh loss nổ (advantage
        #        # lớn -> exp() rất lớn -> mất ổn định huấn luyện)
        #   2. dist = self.actor(observations)   -> phân phối hành động, CÓ gradient
        #   3. log_probs = dist.log_prob(actions)
        #   4. loss = -(exp_adv * log_probs).mean()
        #      (dấu trừ vì ta muốn MAXIMIZE weighted log-likelihood -> minimize -đó)
        dist = ...
        loss = ...

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": loss,
            "mse": torch.mean((dist.mode - actions) ** 2),
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
        metrics_v = self.update_v(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        # HƯỚNG DẪN (Polyak / soft update):
        # target_param <- (1 - tau) * target_param + tau * param,  với tau = self.target_update_rate
        # Cách làm (dùng torch.no_grad() vì đây chỉ là cập nhật giá trị, không phải học bằng gradient):
        #   with torch.no_grad():
        #       for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
        #           target_param.data.mul_(1 - self.target_update_rate)
        #           target_param.data.add_(self.target_update_rate * param.data)
        ...
