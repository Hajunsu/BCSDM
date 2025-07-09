import torch
import torch.nn as nn

Tensor = torch.Tensor

def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unexpected activation: {s_act}")
    
    
class FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_vec, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(
        self,
        out_dim,
        branch_arch,
        trunk_arch,  
    ):
        super(DeepONet, self).__init__()
        
        self.out_dim = out_dim
        self.branch_in_dim = branch_arch["in_dim"]
        self.branch_out_dim = branch_arch["out_dim"]
        self.branch_l_hidden = branch_arch["l_hidden"]
        self.branch_activation = branch_arch["activation"]
        self.brach_out_activation = branch_arch["out_activation"]
        self.trunk_in_dim = trunk_arch["in_dim"]
        self.trunk_out_dim = trunk_arch["out_dim"]
        self.trunk_l_hidden = trunk_arch["l_hidden"]
        self.trunk_activation = trunk_arch["activation"]
        self.trunk_out_activation = trunk_arch["out_activation"]
        
        self.branch_net = FC_vec(
            in_chan=self.branch_in_dim,
            out_chan=self.branch_out_dim * self.out_dim,
            l_hidden=self.branch_l_hidden,
            activation=self.branch_activation,
            out_activation=self.brach_out_activation,
        )
        self.trunck_net = FC_vec(
            in_chan=self.trunk_in_dim,
            out_chan=self.trunk_out_dim * self.out_dim,
            l_hidden=self.trunk_l_hidden,
            activation=self.trunk_activation,
            out_activation=self.trunk_out_activation,
        )
    
    def forward(self, branch_in, trunk_in):
        branch_out = self.branch_net(branch_in).reshape(len(branch_in),-1,self.branch_out_dim)
        trunk_out = self.trunck_net(trunk_in).reshape(len(trunk_in),-1,self.trunk_out_dim)
        return torch.sum(branch_out * trunk_out, dim=-1, keepdim=False)