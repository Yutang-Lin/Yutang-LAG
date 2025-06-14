import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.utils import check


class SACCritic(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(SACCritic, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)

        # (1) feature extraction module for observations
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        
        # (2) feature extraction module for actions
        self.act_base = MLPBase(act_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        
        # (3) rnn module 
        input_size = int(self.base.output_size) + int(self.act_base.output_size)
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = int(self.rnn.output_size)
            
        # (4) Q-function modules (twin Q-functions)
        if len(self.act_hidden_size) > 0:
            self.mlp1 = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
            self.mlp2 = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
            input_size = int(self.act_hidden_size.split(' ')[-1])
        
        self.Q1 = nn.Linear(input_size, 1)
        self.Q2 = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, actions, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # Extract features from observations and actions
        obs_features = self.base(obs)
        act_features = self.act_base(actions)
        
        # Combine features
        critic_features = torch.cat([obs_features, act_features], dim=-1)

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        # Compute Q-values using twin Q-functions
        if len(self.act_hidden_size) > 0:
            Q1_features = self.mlp1(critic_features)
            Q2_features = self.mlp2(critic_features)
            Q1 = self.Q1(Q1_features)
            Q2 = self.Q2(Q2_features)
        else:
            Q1 = self.Q1(critic_features)
            Q2 = self.Q2(critic_features)

        return Q1, Q2, rnn_states
