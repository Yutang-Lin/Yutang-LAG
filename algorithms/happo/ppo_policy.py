import torch
import torch.nn as nn
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr
        self.num_agents = num_agents

        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space
        
        self.actor = nn.ModuleList([PPOActor(args, self.obs_space, self.act_space, self.device) for _ in range(num_agents)])
        self.critic = PPOCritic(args, self.cent_obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, num_agents):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions = []
        action_log_probs = []
        new_rnn_states_actor = []
        for agent_id in range(num_agents):
            action, action_log_prob, rnn_state_actor = self.actor[agent_id](obs[:, agent_id], rnn_states_actor[:, agent_id], masks[:, agent_id])
            actions.append(action)
            action_log_probs.append(action_log_prob)
            new_rnn_states_actor.append(rnn_state_actor)

        def flatten(x):
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        
        actions = flatten(torch.stack(actions, dim=1))
        action_log_probs = flatten(torch.stack(action_log_probs, dim=1))
        rnn_states_actor = flatten(torch.stack(new_rnn_states_actor, dim=1))

        values, rnn_states_critic = self.critic(flatten(cent_obs), flatten(rnn_states_critic), flatten(masks))
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, agent_id, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor[agent_id].evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        return None, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        assert obs.shape[1] == self.num_agents
        
        actions = []
        rnn_states_actors = []
        for agent_id in range(self.num_agents):
            action, _, rnn_state_actor = self.actor[agent_id](obs[:, agent_id], rnn_states_actor[:, agent_id], masks[:, agent_id], deterministic)
            actions.append(action)
            rnn_states_actors.append(rnn_state_actor)

        actions = torch.stack(actions, dim=1).flatten(0, 1)
        rnn_states_actors = torch.stack(rnn_states_actors, dim=1).flatten(0, 1)
        return actions, rnn_states_actors

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
