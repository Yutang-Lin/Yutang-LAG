import torch
import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod
from .utils import get_shape_from_space


class Buffer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def insert(self, **kwargs):
        pass

    @abstractmethod
    def after_update(self):
        pass

    @abstractmethod
    def clear(self):
        pass


class ReplayBuffer(Buffer):

    @staticmethod
    def _flatten(T: int, N: int, x: np.ndarray):
        return x.reshape(T * N, *x.shape[2:])

    @staticmethod
    def _cast(x: np.ndarray):
        return x.transpose(1, 2, 0, *range(3, x.ndim)).reshape(-1, *x.shape[3:])

    def __init__(self, args, num_agents, obs_space, act_space):
        # buffer config
        self.buffer_size = args.buffer_size
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = num_agents
        self.gamma = args.gamma
        self.use_proper_time_limits = args.use_proper_time_limits
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, a_0, r_0, d_1, o_1, ... , d_T, o_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: bad_masks[t] = 'bad_transition' in info[t-1], which indicates whether obs[t] a true terminal state or time limit end state
        self.bad_masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        self.step = 0

    @property
    def advantages(self) -> np.ndarray:
        advantages = self.returns[:-1] - self.value_preds[:-1]  # type: np.ndarray
        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def insert(self,
               obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               **kwargs):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              mask[t+1] = 1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
        """
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.buffer_size

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()

    def clear(self):
        self.step = 0
        self.obs = np.zeros_like(self.obs, dtype=np.float32)
        self.actions = np.zeros_like(self.actions, dtype=np.float32)
        self.rewards = np.zeros_like(self.rewards, dtype=np.float32)
        self.masks = np.ones_like(self.masks, dtype=np.float32)
        self.bad_masks = np.ones_like(self.bad_masks, dtype=np.float32)
        self.action_log_probs = np.zeros_like(self.action_log_probs, dtype=np.float32)
        self.value_preds = np.zeros_like(self.value_preds, dtype=np.float32)
        self.returns = np.zeros_like(self.returns, dtype=np.float32)
        self.rnn_states_actor = np.zeros_like(self.rnn_states_critic)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

    def compute_returns(self, next_value: np.ndarray):
        """
        Compute returns either as discounted sum of rewards, or using GAE.

        Args:
            next_value(np.ndarray): value predictions for the step after the last episode step.
        """
        if self.use_proper_time_limits:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) \
                        * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    @staticmethod
    def recurrent_generator(buffer: Union[Buffer, List[Buffer]], num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            buffers (Buffer or List[Buffer])
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch)
        """
        buffer = [buffer] if isinstance(buffer, ReplayBuffer) else buffer  # type: List[ReplayBuffer]
        n_rollout_threads = buffer[0].n_rollout_threads
        buffer_size = buffer[0].buffer_size
        num_agents = buffer[0].num_agents
        assert all([b.n_rollout_threads == n_rollout_threads for b in buffer]) \
            and all([b.buffer_size == buffer_size for b in buffer]) \
            and all([b.num_agents == num_agents for b in buffer]) \
            and all([isinstance(b, ReplayBuffer) for b in buffer]), \
            "Input buffers must has the same type and shape"
        buffer_size = buffer_size * len(buffer)

        assert n_rollout_threads * buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) * num_agents ({})"
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, buffer_size, num_agents, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = np.vstack([ReplayBuffer._cast(buf.obs[:-1]) for buf in buffer])
        actions = np.vstack([ReplayBuffer._cast(buf.actions) for buf in buffer])
        masks = np.vstack([ReplayBuffer._cast(buf.masks[:-1]) for buf in buffer])
        old_action_log_probs = np.vstack([ReplayBuffer._cast(buf.action_log_probs) for buf in buffer])
        advantages = np.vstack([ReplayBuffer._cast(buf.advantages) for buf in buffer])
        returns = np.vstack([ReplayBuffer._cast(buf.returns[:-1]) for buf in buffer])
        value_preds = np.vstack([ReplayBuffer._cast(buf.value_preds[:-1]) for buf in buffer])
        rnn_states_actor = np.vstack([ReplayBuffer._cast(buf.rnn_states_actor[:-1]) for buf in buffer])
        rnn_states_critic = np.vstack([ReplayBuffer._cast(buf.rnn_states_critic[:-1]) for buf in buffer])

        # Get mini-batch size and shuffle chunk data
        data_chunks = n_rollout_threads * buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            actions_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *buffer[0].rnn_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *buffer[0].rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = ReplayBuffer._flatten(L, N, obs_batch)
            actions_batch = ReplayBuffer._flatten(L, N, actions_batch)
            masks_batch = ReplayBuffer._flatten(L, N, masks_batch)
            old_action_log_probs_batch = ReplayBuffer._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = ReplayBuffer._flatten(L, N, advantages_batch)
            returns_batch = ReplayBuffer._flatten(L, N, returns_batch)
            value_preds_batch = ReplayBuffer._flatten(L, N, value_preds_batch)

            yield obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch


class SharedReplayBuffer(ReplayBuffer):

    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        # env config
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        # buffer config
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.use_proper_time_limits = args.use_proper_time_limits
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        share_obs_shape = get_shape_from_space(share_obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, s_0, a_0, r_0, d_0, ..., o_T, s_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.share_obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state .... same for all agents
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        # NOTE: active_masks[t, :, i] represents whether agent[i] is alive in obs[t] .... differ in different agents
        self.active_masks = np.ones_like(self.masks)
        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        self.step = 0

    def insert(self,
               obs: np.ndarray,
               share_obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               active_masks: Union[np.ndarray, None] = None,
               available_actions: Union[np.ndarray, None] = None):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            share_obs:          s_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
            active_masks:       1 - agent_done_{t}
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            pass
        return super().insert(obs, actions, rewards, masks, action_log_probs, value_preds, rnn_states_actor, rnn_states_critic)

    def after_update(self):
        self.active_masks[0] = self.active_masks[-1].copy()
        self.share_obs[0] = self.share_obs[-1].copy()
        return super().after_update()

    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            advantages (np.ndarray): advantage estimates.
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch)
        """
        assert self.n_rollout_threads * self.buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) "
            "to be greater than or equal to the number of data chunk length ({}).".format(
                self.n_rollout_threads, self.buffer_size, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = self._cast(self.obs[:-1])
        share_obs = self._cast(self.share_obs[:-1])
        actions = self._cast(self.actions)
        masks = self._cast(self.masks[:-1])
        active_masks = self._cast(self.active_masks[:-1])
        old_action_log_probs = self._cast(self.action_log_probs)
        advantages = self._cast(advantages)
        returns = self._cast(self.returns[:-1])
        value_preds = self._cast(self.value_preds[:-1])
        rnn_states_actor = self._cast(self.rnn_states_actor[:-1])
        rnn_states_critic = self._cast(self.rnn_states_critic[:-1])

        # Get mini-batch size and shuffle chunk data
        data_chunks = self.n_rollout_threads * self.buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, NA, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = self._flatten(L, N, obs_batch)
            share_obs_batch = self._flatten(L, N, share_obs_batch)
            actions_batch = self._flatten(L, N, actions_batch)
            masks_batch = self._flatten(L, N, masks_batch)
            active_masks_batch = self._flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = self._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = self._flatten(L, N, advantages_batch)
            returns_batch = self._flatten(L, N, returns_batch)
            value_preds_batch = self._flatten(L, N, value_preds_batch)

            yield obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch


class HAReplayBuffer(SharedReplayBuffer):

    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        super().__init__(args, num_agents, obs_space, share_obs_space, act_space)
        # Add joint rewards and returns
        self.joint_rewards = np.zeros((self.buffer_size, self.n_rollout_threads, 1), dtype=np.float32)
        self.joint_returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.joint_value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, 1), dtype=np.float32)

    @staticmethod
    def _cast(x: np.ndarray):
        return x.transpose(1, 0, *range(2, x.ndim)).reshape(-1, *x.shape[2:])

    def insert(self,
               obs: np.ndarray,
               share_obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               active_masks: Union[np.ndarray, None] = None,
               available_actions: Union[np.ndarray, None] = None):
        # Compute joint rewards (sum of individual rewards)
        self.joint_rewards[self.step] = np.sum(rewards, axis=1)
        return super().insert(obs, share_obs, actions, rewards, masks, action_log_probs, value_preds, 
                            rnn_states_actor, rnn_states_critic, bad_masks, active_masks, available_actions)

    def compute_returns(self, next_value: np.ndarray):
        # Compute joint returns
        if self.use_proper_time_limits:
            if self.use_gae:
                self.joint_value_preds[-1] = np.sum(next_value, axis=1)
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    # Calculate TD error using joint rewards and values
                    td_delta = self.joint_rewards[step] + self.gamma * self.joint_value_preds[step + 1] * np.all(self.masks[step + 1, :, 0:1], axis=1) - self.joint_value_preds[step]
                    # Update GAE using TD error and mask
                    gae = td_delta + self.gamma * self.gae_lambda * np.all(self.masks[step + 1, :, 0:1], axis=1) * gae
                    # Apply bad mask to GAE
                    gae = gae * self.bad_masks[step + 1, 0, 0:1]  # Fixed: Use same indexing as masks
                    # Store returns as GAE + value
                    self.joint_returns[step] = gae + self.joint_value_preds[step]
            else:
                self.joint_returns[-1] = np.sum(next_value, axis=1)
                for step in reversed(range(self.rewards.shape[0])):
                    # Calculate returns using proper time limits
                    self.joint_returns[step] = (self.joint_returns[step + 1] * self.gamma * np.all(self.masks[step + 1, :, 0:1], axis=1) + self.joint_rewards[step]) \
                        * self.bad_masks[step + 1, 0, 0:1] + (1 - self.bad_masks[step + 1, 0, 0:1]) * self.joint_value_preds[step]
        else:
            if self.use_gae:
                self.joint_value_preds[-1] = np.sum(next_value, axis=1)
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    # Calculate TD error using joint rewards and values
                    td_delta = self.joint_rewards[step] + self.gamma * self.joint_value_preds[step + 1] * np.all(self.masks[step + 1, :, 0:1], axis=1) - self.joint_value_preds[step]
                    # Update GAE using TD error and mask
                    gae = td_delta + self.gamma * self.gae_lambda * np.all(self.masks[step + 1, :, 0:1], axis=1) * gae
                    # Store returns as GAE + value
                    self.joint_returns[step] = gae + self.joint_value_preds[step]
            else:
                self.joint_returns[-1] = np.sum(next_value, axis=1)
                for step in reversed(range(self.rewards.shape[0])):
                    # Calculate returns without proper time limits
                    self.joint_returns[step] = self.joint_returns[step + 1] * self.gamma * np.all(self.masks[step + 1, :, 0:1], axis=1) + self.joint_rewards[step]

        # Compute individual returns for value function training
        super().compute_returns(next_value)

    @property
    def advantages(self) -> np.ndarray:
        # Compute joint advantages
        joint_advantages = self.joint_returns[:-1] - self.joint_value_preds[:-1]
        joint_advantages = (joint_advantages - joint_advantages.mean()) / (joint_advantages.std() + 1e-5)
        
        # Expand joint advantages to match individual agent shape
        joint_advantages = np.expand_dims(joint_advantages, axis=2).repeat(self.num_agents, axis=2)
        
        return joint_advantages

    def after_update(self):
        self.joint_returns[0] = self.joint_returns[-1].copy()
        self.joint_value_preds[0] = self.joint_value_preds[-1].copy()
        return super().after_update()

    def clear(self):
        self.joint_rewards = np.zeros_like(self.joint_rewards)
        self.joint_returns = np.zeros_like(self.joint_returns)
        self.joint_value_preds = np.zeros_like(self.joint_value_preds)
        return super().clear()

    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            advantages (np.ndarray): advantage estimates.
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch)
        """
        assert self.n_rollout_threads * self.buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) "
            "to be greater than or equal to the number of data chunk length ({}).".format(
                self.n_rollout_threads, self.buffer_size, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = self._cast(self.obs[:-1])
        share_obs = self._cast(self.share_obs[:-1])
        actions = self._cast(self.actions)
        masks = self._cast(self.masks[:-1])
        active_masks = self._cast(self.active_masks[:-1])
        old_action_log_probs = self._cast(self.action_log_probs)
        advantages = self._cast(advantages)
        returns = self._cast(self.joint_returns[:-1])
        value_preds = self._cast(self.joint_value_preds[:-1])
        rnn_states_actor = self._cast(self.rnn_states_actor[:-1])
        rnn_states_critic = self._cast(self.rnn_states_critic[:-1])

        # Get mini-batch size and shuffle chunk data
        data_chunks = self.n_rollout_threads * self.buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, NA, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states_actor.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = self._flatten(L, N, obs_batch)
            share_obs_batch = self._flatten(L, N, share_obs_batch)
            actions_batch = self._flatten(L, N, actions_batch)
            masks_batch = self._flatten(L, N, masks_batch)
            active_masks_batch = self._flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = self._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = self._flatten(L, N, advantages_batch)
            returns_batch = self._flatten(L, N, returns_batch)
            value_preds_batch = self._flatten(L, N, value_preds_batch)

            yield obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch


class HAQReplayBuffer(HAReplayBuffer):

    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        super().__init__(args, num_agents, obs_space, share_obs_space, act_space)

    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            advantages (np.ndarray): advantage estimates.
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch, rewards_batch, next_obs_batch, next_share_obs_batch)
        """
        assert self.n_rollout_threads * self.buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) "
            "to be greater than or equal to the number of data chunk length ({}).".format(
                self.n_rollout_threads, self.buffer_size, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = self._cast(self.obs[:-1])
        share_obs = self._cast(self.share_obs[:-1])
        next_obs = self._cast(self.obs[1:])
        next_share_obs = self._cast(self.share_obs[1:])
        rewards = self._cast(self.rewards)
        joint_rewards = self._cast(self.joint_rewards)
        actions = self._cast(self.actions)
        masks = self._cast(self.masks[:-1])
        active_masks = self._cast(self.active_masks[:-1])
        old_action_log_probs = self._cast(self.action_log_probs)
        advantages = self._cast(advantages)
        returns = self._cast(self.returns[:-1])
        joint_returns = self._cast(self.joint_returns[:-1])
        value_preds = self._cast(self.value_preds[:-1])
        joint_value_preds = self._cast(self.joint_value_preds[:-1])
        rnn_states_actor = self._cast(self.rnn_states_actor[:-1])
        rnn_states_critic = self._cast(self.rnn_states_critic[:-1])

        # Get mini-batch size and shuffle chunk data
        data_chunks = self.n_rollout_threads * self.buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            next_obs_batch = []
            share_obs_batch = []
            next_share_obs_batch = []
            actions_batch = []
            rewards_batch = []
            joint_rewards_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            joint_returns_batch = []
            value_preds_batch = []
            joint_value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                next_obs_batch.append(next_obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                next_share_obs_batch.append(next_share_obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                rewards_batch.append(rewards[ind:ind + data_chunk_length])
                joint_rewards_batch.append(joint_rewards[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                joint_returns_batch.append(joint_returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                joint_value_preds_batch.append(joint_value_preds[ind:ind + data_chunk_length])
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, NA, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            next_obs_batch = np.stack(next_obs_batch, axis=1)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            next_share_obs_batch = np.stack(next_share_obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            rewards_batch = np.stack(rewards_batch, axis=1)
            joint_rewards_batch = np.stack(joint_rewards_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            joint_returns_batch = np.stack(joint_returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            joint_value_preds_batch = np.stack(joint_value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states_actor.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = self._flatten(L, N, obs_batch)
            next_obs_batch = self._flatten(L, N, next_obs_batch)
            share_obs_batch = self._flatten(L, N, share_obs_batch)
            next_share_obs_batch = self._flatten(L, N, next_share_obs_batch)
            actions_batch = self._flatten(L, N, actions_batch)
            rewards_batch = self._flatten(L, N, rewards_batch)
            joint_rewards_batch = self._flatten(L, N, joint_rewards_batch)
            masks_batch = self._flatten(L, N, masks_batch)
            active_masks_batch = self._flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = self._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = self._flatten(L, N, advantages_batch)
            returns_batch = self._flatten(L, N, returns_batch)
            joint_returns_batch = self._flatten(L, N, joint_returns_batch)
            value_preds_batch = self._flatten(L, N, value_preds_batch)
            joint_value_preds_batch = self._flatten(L, N, joint_value_preds_batch)

            num_agents = obs_batch.shape[1]
            extend_joint_rewards_batch = np.repeat(joint_rewards_batch, num_agents, axis=1)

            yield obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch, extend_joint_rewards_batch, next_obs_batch, next_share_obs_batch