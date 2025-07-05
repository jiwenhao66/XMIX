import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class XMIX(nn.Module):
    def __init__(self, scheme, input_shape, args):
        super(XMIX, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.embed_dim = args.mixing_embed_dim
        self.att_out_dim = args.att_out_dim
        if args.env == 'sc2_v2':
            self.n_enemies = args.n_enemies
            self.n_entities = self.n_agents + self.n_enemies
        else:
            self.n_entities = args.n_entities
        if args.env == 'sc2_v2':
            self.feat_dim = input_shape // self.n_entities
        else:
            self.feat_dim = args.state_entity_feats
        self.state_dim = int(np.prod(args.state_shape))
        self.rnn_hidden_dim = self.args.rnn_hidden_dim
        self.abs = getattr(self.args, 'abs', True)
        self.attention_dim = args.attention_dim
        self.scale_factor = nn.Parameter(th.tensor(1.0))
        self.factor = nn.Parameter(th.tensor(1.0))
        
        self.token_embedding = nn.Linear(self.feat_dim, args.mix_emb_dim)
        self.y_fc = nn.Linear(args.mix_emb_dim, self.feat_dim) 
        hypernet_embed = self.args.hypernet_embed
        
        self.abs = getattr(self.args, 'abs', True)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
    def forward(self, agent_qs, hist, states, obs):
        bs = agent_qs.size(0)
        t = states.size(1)
        states = states.reshape(-1, self.state_dim)
        s = states.reshape(bs, t, self.n_entities, self.feat_dim)
        self.state_shape = s.size()
        states = s.reshape(-1, self.n_entities, self.feat_dim)
        states = F.relu(self.token_embedding(states))
        y = self.transformer.forward(states, hist)[:, :self.args.n_agents]
        y =  F.relu(self.y_fc(y))
        y = y.reshape(-1, self.state_dim)

        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(y).abs() if self.abs else self.hyper_w_1(y)
        b1 = self.hyper_b_1(y)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = self.hyper_w_final(y).abs() if self.abs else self.hyper_w_final(y)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(y).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        hidden = F.elu(th.bmm(agent_qs.clone().detach(), w1) + b1)
        delu = th.ones_like(hidden)
        mask = hidden < 0
        delu[mask] = th.exp(hidden[mask])
        grad = th.bmm(w1, th.diag_embed(delu.squeeze(1)))
        grad = th.bmm(grad, w_final)
        grad = grad.reshape(bs, -1, self.n_agents)
        return q_tot, grad
