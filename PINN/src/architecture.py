import torch
import torch.nn as nn
from torch.profiler import record_function


# Usage example (d=15 case)
# input_dim = 15 + 1
# ff = FourierFeatures(input_dim, num_freqs=320, sigma=30.0)
# or multi-scale:
# ff = FourierFeatures(input_dim, num_freqs=300, scale_multiples=[1, 10, 50])
class FourierFeatures(nn.Module):
    def __init__(self, D, num_freqs=256, sigma=20.0, scale_multiples=None):
        super().__init__()
        self.num_freqs = num_freqs
        self.sigma = sigma
        self.input_dim = D
        self.ouput_dim = 2 * num_freqs

        # Optional: multi-scale
        self.multi_scale = scale_multiples is not None
        if self.multi_scale:
            self.scales = scale_multiples
            self.Bs = nn.ParameterList()
            for s in scale_multiples:
                Bi = torch.randn(num_freqs//len(scale_multiples), D) * s
                self.Bs.append(nn.Parameter(Bi, requires_grad=False))  # still fixed
        else:
            # === Sample B once and freeze it ===
            B = torch.randn(num_freqs, D) * sigma   # Normal(0, sigma²)
            self.register_buffer('B', B)                    # fixed, not in optimizer

    def forward(self, z):  # z: (batch, d+1)
        if self.multi_scale:
            feats = []
            for i, Bi in enumerate(self.Bs):
                x = 2 * torch.pi * (z @ Bi.T)
                feats.append(torch.cat([torch.cos(x), torch.sin(x)], dim=-1))
            return torch.cat(feats, dim=-1)
        else:
            x = 2 * torch.pi * (z @ self.B.T)          # (batch, M)
            return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class ResNetBlock(nn.Module):
    def __init__(self, width, activation=nn.Mish):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            activation(),
            nn.Linear(width, width)
        )
        self.act = activation()

    def forward(self, x):
        residual = x
        out = self.block(x)
        return self.act(out + residual)




class PINN(nn.Module):
    def __init__(self, D, layers=[64], activation_fn=nn.Tanh, ff=None):
        """
        D: input dimension (d spatial dims + 1 time dim)
        activation_fn: 'nn.Tanh', 'nn.SiLU'
        """
        super().__init__()

        net_layers = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            net_layers.append(nn.Linear(l1,l2))
            net_layers.append(activation_fn())

        self.ff = ff
        if ff:
            first = nn.Linear(ff.ouput_dim, layers[0])
        else:
            first = nn.Linear(D, layers[0])
        self.net = nn.Sequential(
            first, activation_fn(),
            *net_layers,
            nn.Linear(layers[-1], 1)
        )

    def forward(self, X):
        with record_function("forward"):
            if self.ff:
                return self.net(self.ff(X))
            else:
                return self.net(X)


#d=5:  num_freqs=128, sigma=8, hidden_width=128, num_blocks=4
#d=10: num_freqs=256, sigma=15, hidden_width=256, num_blocks=5
#d=15: num_freqs=320, sigma=30, hidden_width=256, num_blocks=6
#d=20: num_freqs=512, sigma=40, hidden_width=256, num_blocks=8
class ResPINN(nn.Module):
    def __init__(self, D, hidden_width=256, num_blocks=5, ff=None):
        super().__init__()
        
        self.ff = ff
        if ff:
            self.first = nn.Linear(ff.ouput_dim, hidden_width)
        else:
            self.first = nn.Linear(D, hidden_width)
        
        # Stack residual blocks
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_width) for _ in range(num_blocks)
        ])
        
        # Final layer
        self.out = nn.Linear(hidden_width, 1)
        
    def forward(self, X):
        with record_function("forward"):
            if self.ff:
                h = self.first(self.ff(X))
            else:
                h = self.first(X)
            h = nn.Mish()(h)
        
            for block in self.blocks:
                h = block(h)
        
            return self.out(h)



#####################################################################
#####################################################################
#####################################################################
#####################################################################


class PINN_SepTime(nn.Module):
    def __init__(self, D, layers=[64], activation_fn=nn.Tanh):
        """
        D: input dimension (d spatial dims + 1 time dim)
        activation_fn: 'nn.Tanh', 'nn.SiLU'
        """
        super().__init__()

        # spatial termporal
        net_layers = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            net_layers.append(nn.Linear(l1,l2))
            net_layers.append(activation_fn())
        self.net_sptemp = nn.Sequential(
            nn.Linear(D, layers[0]), activation_fn(),
            *net_layers,
            nn.Linear(layers[-1], 1)
        )
        # temporal
        self.net_temp = nn.Sequential(
            nn.Linear(1, 128), activation_fn(),
            nn.Linear(128, 64), activation_fn(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        with record_function("forward"):
            return self.net_sptemp(X) * self.net_temp(X[:,-1:])

class PINN_SeparableTimes(nn.Module):
    def __init__(self, D, layers=[64], activation_fn=nn.Tanh):
        """
        D: input dimension (d spatial dims + 1 time dim)
        activation_fn: 'nn.Tanh', 'nn.SiLU'
        """
        super().__init__()
        self.d = D-1

        self.nets_1dspatial_temporal = []
        for di in range(self.d):
            net_layers = []
            for l1,l2 in zip(layers[:-1], layers[1:]):
                net_layers.append(nn.Linear(l1,l2))
                net_layers.append(activation_fn())
            self.nets_1dspatial_temporal.append( nn.Sequential(
                nn.Linear(2, layers[0]), activation_fn(),
                *net_layers,
                nn.Linear(layers[-1], 1)
            ) )
        self.net_temporal = nn.Sequential(
            nn.Linear(1, 128), activation_fn(),
            nn.Linear(128, 64), activation_fn(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        with record_function("forward"):
            out = torch.ones((X.shape[0],1))
            for di in range(self.d):
                out *= self.nets_1dspatial_temporal[di](torch.cat([X[:,di:di+1],X[:,-1:]],dim=1))
            out *= self.net_temporal(X[:,-1:])
            return out

