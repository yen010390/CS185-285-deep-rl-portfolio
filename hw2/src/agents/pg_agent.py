from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # step 2: calculate advantages from Q values
        # NOTE: _estimate_advantage needs the per-trajectory `rewards` (for GAE,
        # which uses terminals to mark episode boundaries), so we estimate the
        # advantages BEFORE flattening the rewards.
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # flatten the lists of per-trajectory arrays into single arrays with a
        # leading dimension of `batch_size`, so the rest of the code is vectorized.
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)
        # `advantages` is already a flat array returned by _estimate_advantage.

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            critic_info: dict = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        T = len(rewards)
        discounts = self.gamma ** np.arange(T)
        discounted_sum = float(np.sum(discounts * rewards))
        # Every timestep gets the same full-trajectory discounted return.
        return [discounted_sum] * T

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        T = len(rewards)
        rtg = np.zeros(T, dtype=np.float32)
        running = 0.0
        # Walk backwards so each step reuses the next: rtg[t] = r[t] + gamma * rtg[t+1].
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            rtg[t] = running
        return rtg

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # Without a baseline, the advantage is just the Monte Carlo Q-value.
            # q_values is still a list of per-trajectory arrays here -> flatten.
            advantages = np.concatenate(q_values)
        else:
            # Flatten obs/q_values so we can run the critic over the whole batch.
            obs_flat = np.concatenate(obs) if isinstance(obs, (list, tuple)) else obs
            q_values_flat = np.concatenate(q_values)

            # Run the critic to get V(s_t) for every state in the batch.
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs_flat)))
            assert values.shape == q_values_flat.shape

            if self.gae_lambda is None:
                # Baseline but no GAE: A(s,a) = Q(s,a) - V(s).
                advantages = q_values_flat - values
            else:
                # Generalized Advantage Estimation.
                # Flatten rewards/terminals to index per-step.
                rewards_flat = np.concatenate(rewards)
                terminals_flat = (
                    np.concatenate(terminals)
                    if isinstance(terminals, (list, tuple))
                    else terminals
                )
                batch_size = obs_flat.shape[0]

                # Append a dummy V_{T+1}=0 for the recursion.
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # nonterminal = 0 at the last step of a trajectory, so both
                    # the bootstrap V(s_{t+1}) and the recursive A_{t+1} are cut
                    # off at episode boundaries.
                    nonterminal = 1.0 - terminals_flat[i]
                    delta = (
                        rewards_flat[i]
                        + self.gamma * values[i + 1] * nonterminal
                        - values[i]
                    )
                    advantages[i] = (
                        delta
                        + self.gamma * self.gae_lambda * nonterminal * advantages[i + 1]
                    )

                # remove dummy advantage
                advantages = advantages[:-1]

        # normalize the advantages to mean 0 / std 1 within the batch.
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages