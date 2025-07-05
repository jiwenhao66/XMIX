import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
from torch_geometric.data import Data
class XRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(XRNNAgent, self).__init__()
        self.args = args
        self.model = MultiAgentGAT(in_channels=input_shape, out_channels=input_shape, num_heads=4)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(2*args.rnn_hidden_dim, args.n_actions)
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = args.obs_shape
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = self.args.hypernet_embed
        self.fc_out = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.enc_obs = True
        obs_dim = args.rnn_hidden_dim
        if self.enc_obs:
            self.obs_enc_dim = 64
            self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, self.obs_enc_dim),
                                             nn.ReLU())
            self.obs_dim_effective = self.obs_enc_dim
        else:
            self.obs_encoder = nn.Sequential()
            self.obs_dim_effective = obs_dim
            
        self.W_attn_query = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)
        self.W_attn_key = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)
        self.scale_factor = nn.Parameter(th.tensor(1.0))
    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_states=None):
        b, a, e = inputs.size()
        input0 = inputs
        alive_agents = 1. * (th.sum(input0, dim=2) > 0).view(-1, self.n_agents) 
        alive_agents_mask = th.bmm(alive_agents.unsqueeze(2), alive_agents.unsqueeze(1))
        x = F.relu(self.fc1(input0.view(-1, e)), inplace=True)
        if hidden_states is not None:
            hidden_states = hidden_states.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_states)

        encoded_hidden_states = self.obs_encoder(h)
        encoded_hidden_states = encoded_hidden_states.contiguous().view(-1, self.n_agents, self.obs_dim_effective)
        attn_query = self.W_attn_query(encoded_hidden_states)
        attn_key = self.W_attn_key(encoded_hidden_states)
        attn = th.matmul(attn_query, th.transpose(attn_key, 1, 2)) / np.sqrt(self.obs_dim_effective)
        attn = nn.Softmax(dim=2)(attn + (-1e10 * (1 - alive_agents_mask)))
        batch_adj = attn * alive_agents_mask
        out, _, _ = self.mixing_GNN(input0, batch_adj, self.n_agents)
        out = out.reshape(-1, e)
        out = F.relu(self.fc_out(out.view(-1, e)), inplace=True)
        z = out
        out =  th.cat((out, h), 1)
        
        q = self.fc2(out)
        return q.view(b, a, -1), h.view(b, a, -1), z.view(-1, self.n_agents, self.rnn_hidden_dim)
        
        
        
        
        
        
