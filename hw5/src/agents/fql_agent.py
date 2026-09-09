from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


# =============================================================================
# TỔNG QUAN FQL (Flow Q-Learning) — đọc trước khi làm các TODO bên dưới
# =============================================================================
# FQL dùng 2 actor khác nhau:
#   - self.bc_actor (VectorFieldPolicy): học flow-matching behavior cloning.
#     Đây là 1 "vector field" v_theta(s, x, t) dự đoán VẬN TỐC tại điểm x, thời
#     điểm t, biết state s. Để lấy action từ nó, ta phải tích phân ODE
#     dx/dt = v_theta(s, x, t) từ x_0 = noise ~ N(0, I) đến x_1 = action, chia
#     làm `self.flow_steps` bước Euler (chậm vì cần nhiều lần forward).
#   - self.onestep_actor (VectorFieldPolicy, nhưng dùng khác cách): là 1 policy
#     được "chưng cất" (distill) để bắt chước bc_actor NHƯNG chỉ cần 1 lần
#     forward duy nhất: action = self.onestep_actor(obs, noise) (không tích phân
#     ODE, không cần thời gian t).
# Lý do có 2 actor: bc_actor học tốt phân phối dữ liệu nhưng chậm khi inference
# (cần flow_steps lần forward); onestep_actor nhanh (1 lần forward) và còn được
# train thêm để tối đa hoá Q, không chỉ bắt chước dữ liệu.
# =============================================================================


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        # HƯỚNG DẪN:
        # Dùng onestep_actor để lấy action nhanh (1 lần forward, không cần tích phân ODE):
        #   1. Lấy noise ngẫu nhiên: noise = torch.randn((1, self.action_dim), device=observation.device)
        #      (chú ý: batch size ở đây là 1, vì observation đã được thêm chiều batch ở dòng trên)
        #   2. action = self.onestep_actor(observation, noise)
        #      (VectorFieldPolicy.forward(obs, acs, times=None) — nếu không truyền `times`,
        #      nó mặc định = 0, phù hợp vì onestep_actor không dùng khái niệm thời gian
        #      như bc_actor, nó chỉ ánh xạ trực tiếp noise -> action)
        action = ...
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        # HƯỚNG DẪN (Euler integration của ODE dx/dt = v_theta(s, x, t)):
        # Chia đoạn [0, 1] thành `self.flow_steps` bước đều nhau, mỗi bước có
        # độ dài dt = 1 / self.flow_steps. Bắt đầu từ x = noise (tại t=0), lặp:
        #   x_{k+1} = x_k + v_theta(s, x_k, t_k) * dt,  với t_k = k * dt
        # Code mẫu (điền vào):
        #   dt = 1.0 / self.flow_steps
        #   x = noise
        #   for i in range(self.flow_steps):
        #       t = torch.full((*x.shape[:-1], 1), i * dt, device=x.device, dtype=x.dtype)
        #       v = self.bc_actor(observation, x, t)
        #       x = x + v * dt
        #   action = x
        # (Lưu ý: hàm này có @torch.compile, vòng for với self.flow_steps là số
        # nguyên Python cố định nên compile vẫn hoạt động bình thường — không cần lo.)
        action = ...
        action = torch.clamp(action, -1, 1)
        return action

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
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        # HƯỚNG DẪN:
        #   1. Với torch.no_grad():
        #        a. Lấy noise cho next state: next_noise = torch.randn(
        #             (next_observations.shape[0], self.action_dim), device=next_observations.device)
        #        b. next_actions = self.onestep_actor(next_observations, next_noise)
        #        c. Clamp: next_actions = torch.clamp(next_actions, -1, 1)
        #        d. next_q = self.target_critic(next_observations, next_actions).min(dim=0).values
        #        e. target = rewards + self.discount * (1 - dones) * next_q
        #   2. q = self.critic(observations, actions)   -> (n_ensembles, B), CÓ gradient
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
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        # HƯỚNG DẪN (Conditional Flow Matching loss):
        # Ta định nghĩa 1 đường đi tuyến tính (linear interpolation path) từ noise
        # x_0 ~ N(0, I) đến action thật x_1 = actions, tại thời điểm t ngẫu nhiên
        # trong [0, 1]:
        #   x_t = (1 - t) * x_0 + t * x_1
        # Vận tốc "đúng" dọc theo đường thẳng này luôn là hằng số:
        #   dx_t/dt = x_1 - x_0 = actions - noise
        # Ta train bc_actor (vector field) dự đoán đúng vận tốc này tại (s, x_t, t):
        #   1. noise = torch.randn_like(actions)
        #   2. t = torch.rand((*actions.shape[:-1], 1), device=actions.device, dtype=actions.dtype)
        #   3. x_t = (1 - t) * noise + t * actions
        #   4. target_v = actions - noise
        #   5. pred_v = self.bc_actor(observations, x_t, t)
        #   6. loss = ((pred_v - target_v) ** 2).mean()
        loss = ...

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # HƯỚNG DẪN:
        # onestep_actor được train với 2 mục tiêu cộng lại (distillation + Q maximization):
        #
        # (a) Distillation: bắt onestep_actor bắt chước action mà bc_actor (đã tích
        #     phân đầy đủ qua flow_steps bước) sinh ra, TỪ CÙNG 1 noise:
        #       noise = torch.randn((observations.shape[0], self.action_dim), device=observations.device)
        #       with torch.no_grad():
        #           bc_actions = self.get_bc_action(observations, noise)   # target, không cần gradient
        #       onestep_actions = self.onestep_actor(observations, noise)  # CÓ gradient
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        #       distill_loss = ((onestep_actions - bc_actions) ** 2).mean()
        #       (Không clip onestep_actions ở bước này để gradient không bị "cắt cụt"
        #       bởi hàm clamp không khả vi ở biên -1/1.)
        distill_loss = ...

        # (b) Q maximization: onestep_actor còn phải chọn action tối đa hoá Q, giống
        #     phần actor loss trong SAC+BC.
        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        #       clipped_actions = torch.clamp(onestep_actions, -1, 1)
        #       q = self.critic(observations, clipped_actions).min(dim=0).values
        #       q_loss = -self.alpha * q.mean()
        #       (self.alpha đóng vai trò trọng số cân bằng giữa "giống BC" và
        #       "tối đa hoá Q", giống với TD3+BC/AWR ở các agent trước.)
        q_loss = ...

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        # HƯỚNG DẪN: chỉ để log, không cần gradient — có thể .detach():
        #   mse = torch.mean((onestep_actions.detach() - actions) ** 2)
        mse = ...

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
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
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        # HƯỚNG DẪN: giống hệt IQL/SAC+BC — soft update:
        #   target_param <- (1 - tau) * target_param + tau * param, tau = self.target_update_rate
        #   with torch.no_grad():
        #       for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
        #           target_param.data.mul_(1 - self.target_update_rate)
        #           target_param.data.add_(self.target_update_rate * param.data)
        ...
