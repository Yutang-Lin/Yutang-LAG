import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Union, List
from .sac_policy import SACPolicy
from ..utils.buffer import SharedReplayBuffer
from ..utils.utils import check, get_gard_norm


class SACTrainer():
    def __init__(self, args, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # SAC hyperparameters
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # target network update rate
        self.alpha = args.alpha  # entropy regularization coefficient
        self.target_update_interval = args.target_update_interval  # how often to update target network
        self.automatic_entropy_tuning = args.automatic_entropy_tuning  # whether to automatically tune entropy
        self.num_mini_batch = args.num_mini_batch
        # self.target_entropy = -args.act_dim  # target entropy = -dim(A)
        
        # Initialize log_alpha
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.lr)
        
        # network configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm

    def sac_update(self, policy: SACPolicy, sample):
        num_agents = policy.num_agents
        agent_order = np.random.permutation(num_agents)

        obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch, rewards_batch, next_obs_batch, next_share_obs_batch = sample

        # Convert to tensor
        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        rnn_states_actor_batch = check(rnn_states_actor_batch).to(**self.tpdv)
        rnn_states_critic_batch = check(rnn_states_critic_batch).to(**self.tpdv)
        next_share_obs_batch = check(next_share_obs_batch).to(**self.tpdv)
        next_obs_batch = check(next_obs_batch).to(**self.tpdv)

        batch_size = share_obs_batch.shape[0]

        # Update Q-functions
        _, current_Q1, current_Q2, _ = policy.get_values(share_obs_batch, actions_batch, rnn_states_critic_batch, masks_batch, return_q1_q2=True)
        current_Q1 = current_Q1.view(batch_size, -1, 1)
        current_Q2 = current_Q2.view(batch_size, -1, 1)
        
        with torch.no_grad():
            # Get next actions and log probs
            next_actions = []
            next_log_probs = []
            for agent_id in range(num_agents):
                next_action, next_log_prob, _ = policy.actor[agent_id](
                    next_obs_batch[:, agent_id], 
                    rnn_states_actor_batch[:, agent_id],
                    masks_batch[:, agent_id]
                )
                next_actions.append(next_action)
                next_log_probs.append(next_log_prob)
            
            next_actions = torch.stack(next_actions, dim=1)
            next_log_probs = torch.stack(next_log_probs, dim=1)
            
            # Compute target Q-values
            _, target_Q1, target_Q2, _ = policy.get_values(next_share_obs_batch, next_actions, rnn_states_critic_batch, masks_batch, return_q1_q2=True)
            target_Q = torch.min(target_Q1, target_Q2).reshape(batch_size, -1, 1)
            rewards_batch = rewards_batch.reshape(batch_size, -1, 1)

            target_Q = rewards_batch + self.gamma * (1 - masks_batch) * (target_Q - self.alpha * next_log_probs)

        # Compute Q-function loss
        critic_loss = 0.5 * (current_Q1 - target_Q).pow(2).mean() + 0.5 * (current_Q2 - target_Q).pow(2).mean()

        # Update critic first
        policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.critic_optimizer.step()

        # Update policy
        all_policy_loss = 0
        for agent_id in agent_order:
            actions, log_probs, _ = policy.actor[agent_id](
                obs_batch[:, agent_id],
                rnn_states_actor_batch[:, agent_id],
                masks_batch[:, agent_id]
            )
            agent_mask = torch.zeros(1, num_agents, 1).to(**self.tpdv)
            actions = actions[:, None] * agent_mask + (1 - agent_mask) * actions_batch
            
            _, Q1, Q2, _ = policy.get_values(share_obs_batch, actions, rnn_states_critic_batch, masks_batch, return_q1_q2=True)
            Q = torch.min(Q1, Q2).view(batch_size, -1, 1)[:, agent_id, :]
            
            policy_loss = (self.alpha * log_probs - Q).mean()
            all_policy_loss += policy_loss

        # Update actor
        policy.actor_optimizer.zero_grad()
        all_policy_loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
        policy.actor_optimizer.step()

        # Update target networks
        if policy.update_step % self.target_update_interval == 0:
            for param, target_param in zip(policy.critic.parameters(), policy.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, all_policy_loss, actor_grad_norm, critic_grad_norm

    def train(self, policy: SACPolicy, buffer: SharedReplayBuffer):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0

        if self.use_recurrent_policy:
            data_generator = buffer.recurrent_generator(buffer.advantages, self.num_mini_batch, self.data_chunk_length)
        else:
            raise NotImplementedError

        for sample in data_generator:
            value_loss, policy_loss, actor_grad_norm, critic_grad_norm = self.sac_update(policy, sample)

            train_info['value_loss'] += value_loss.item()
            train_info['policy_loss'] += policy_loss.item()
            train_info['actor_grad_norm'] += actor_grad_norm
            train_info['critic_grad_norm'] += critic_grad_norm

        num_updates = self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
