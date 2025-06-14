import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import SharedReplayBuffer
from ..utils.utils import check, get_gard_norm


class PPOTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length

    def ppo_update(self, policy: PPOPolicy, sample):
        num_agents = policy.num_agents
        agent_order = np.random.permutation(num_agents)

        obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        rnn_states_critic_batch = check(rnn_states_critic_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        all_policy_loss = 0
        for agent_id in agent_order:
            _, action_log_probs, dist_entropy = policy.evaluate_actions(share_obs_batch,
                                                                         obs_batch[:, agent_id],
                                                                         rnn_states_actor_batch[:, agent_id],
                                                                         rnn_states_critic_batch,
                                                                         actions_batch[:, agent_id],
                                                                         masks_batch[:, agent_id],
                                                                         agent_id)
            # Obtain the loss function
            ratio = torch.exp(action_log_probs - old_action_log_probs_batch[:, agent_id])
            surr1 = ratio * advantages_batch[:, agent_id]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch[:, agent_id]
            policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
            policy_loss = -policy_loss.mean()
            all_policy_loss += policy_loss
        
        values = policy.get_values(share_obs_batch.flatten(0, 1),
                                   rnn_states_critic_batch.flatten(0, 1),
                                   masks_batch.flatten(0, 1))
        if values.shape != returns_batch.shape:
            returns_batch = returns_batch.repeat(1, num_agents).reshape(values.shape)
            value_preds_batch = value_preds_batch.repeat(1, num_agents).reshape(values.shape)

        returns_batch = returns_batch.view(-1)
        value_preds_batch = value_preds_batch.view(-1)
        values = values.view(-1)

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()

        loss = all_policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

        # Optimize the loss function
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: SharedReplayBuffer):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.recurrent_generator(buffer.advantages, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
