import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

from .sac_actor import SACActor
from .sac_critic import SACCritic
from ..utils.utils import check


class SACPolicy:
    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space
        
        # Calculate total action dimension across all agents
        self.total_act_dim = act_space.shape[0] * num_agents

        # Create a proper gym space for the critic's action input
        self.critic_act_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.total_act_dim,),
            dtype=np.float32
        )

        # Initialize actor networks for each agent
        self.actor = nn.ModuleList([
            SACActor(args, obs_space, act_space, device) for _ in range(num_agents)
        ])
        
        # Initialize critic network (shared across agents)
        self.critic = SACCritic(args, cent_obs_space, self.critic_act_space, device)
        self.critic_target = SACCritic(args, cent_obs_space, self.critic_act_space, device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)
        
        # Initialize temperature parameter
        self.alpha = args.alpha
        if args.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr)
        
        self.update_step = 0

    @torch.no_grad()
    def get_actions(self, shared_obs, obs, rnn_states, rnn_states_critic, masks, num_agents, deterministic=False):
        actions = []
        action_log_probs = []
        rnn_states_actor = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = self.actor[agent_id](
                obs[:, agent_id], 
                rnn_states[:, agent_id],
                masks[:, agent_id],
                deterministic
            )
            actions.append(action)
            action_log_probs.append(action_log_prob)
            rnn_states_actor.append(rnn_state)
            
        actions = torch.stack(actions, dim=1)
        action_log_probs = torch.stack(action_log_probs, dim=1).flatten(0, 1)
        rnn_states_actor = torch.stack(rnn_states_actor, dim=1).flatten(0, 1)

        # Get Q-values for the actions
        values, rnn_states_critic = self.get_values(shared_obs, actions, rnn_states_critic, masks, return_rnn_states=True)
        return values, actions.flatten(0, 1), action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, actions, rnn_states, masks, return_rnn_states=False, return_q1_q2=False):
        def flatten(x):
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        
        # Flatten actions across all agents
        batch_size = actions.shape[0]
        actions = actions.reshape(batch_size, 1, -1)

        actions = check(actions).to(**self.tpdv)
        actions = actions.repeat(1, self.num_agents, 1).flatten(0, 1)
        
        q1, q2, rnn_states_critic = self.critic(flatten(cent_obs), actions, flatten(rnn_states), flatten(masks))
        if return_q1_q2:
            return torch.min(q1, q2), q1, q2, rnn_states_critic
        else:
            if return_rnn_states:
                return torch.min(q1, q2), rnn_states_critic
            else:
                return torch.min(q1, q2)

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        action_log_probs = []
        dist_entropy = []
        
        for agent_id in range(self.num_agents):
            action_log_prob, entropy = self.actor[agent_id].evaluate_actions(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                action[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id] if active_masks is not None else None
            )
            action_log_probs.append(action_log_prob)
            dist_entropy.append(entropy)
            
        action_log_probs = torch.stack(action_log_probs, dim=1)
        dist_entropy = torch.stack(dist_entropy, dim=1).mean()
        
        return action_log_probs, dist_entropy

    def act(self, obs, rnn_states, masks, deterministic=False):
        assert obs.shape[1] == self.num_agents

        actions = []
        rnn_states_actors = []
        for agent_id in range(self.num_agents):
            action, _, rnn_state = self.actor[agent_id](obs[:, agent_id], rnn_states[:, agent_id], masks[:, agent_id], deterministic)
            actions.append(action)
            rnn_states_actors.append(rnn_state)

        actions = torch.stack(actions, dim=1).flatten(0, 1)
        rnn_states_actors = torch.stack(rnn_states_actors, dim=1).flatten(0, 1)
        return actions, rnn_states_actors

    def prep_training(self):
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def copy(self):
        return SACPolicy(self.args, self.num_agents, self.obs_space, self.cent_obs_space, self.act_space, self.device)
