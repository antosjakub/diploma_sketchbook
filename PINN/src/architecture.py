import torch
import torch.nn as nn
from torch.profiler import record_function
import utility


# --------------- Random Weight Factorization (Sec 4.3, arXiv:2308.08468) ---------------
# W = diag(exp(s)) @ V.  Initialised so that the effective W matches Glorot.
# Forward: y = (x @ V^T) * exp(s) + b  — no full W materialised.
class RWFLinear(nn.Module):
    def __init__(self, in_features, out_features, mu=1.0, sigma=0.1):
        super().__init__()
        W = torch.empty(out_features, in_features)
        nn.init.xavier_normal_(W)
        s = torch.randn(out_features) * sigma + mu
        V = torch.exp(-s).unsqueeze(1) * W        # so diag(exp(s)) @ V == W
        self.s = nn.Parameter(s)
        self.V = nn.Parameter(V)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.V.t() * torch.exp(self.s) + self.bias


def _linear(input_dim, output_dim, rwf={}):
    if rwf:
        return RWFLinear(input_dim, output_dim, **rwf)
    else:
        return nn.Linear(input_dim, output_dim)


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
        self.output_dim = 2 * num_freqs

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
    def __init__(self, width, activation=nn.Mish, rwf={}):
        super().__init__()
        self.l1 = _linear(width, width, **rwf)
        self.l2 = _linear(width, width, **rwf)
        self.act = activation()

    def forward(self, x):
        return self.act(self.l2(self.act(self.l1(x))) + x)



class PINN(nn.Module):
    def __init__(self, input_dim, layers=[64], output_dim=1, activation_fn=nn.Tanh, head_fn=utility.identity_fn,
            ff=None,
            modified_mlp=False,
            rwf={}
        ):
        f"""
        D: input dimension (d spatial dims + 1 time dim)
        activation_fn: 'nn.Tanh', 'nn.SiLU'
        modified_mlp: use Modified MLP with input-gated hidden layers (Sec 6.4, arXiv:2308.08468)
        rwf: dict['mu': 1.0, 'sigma': 0.1] use Random Weight Factorization on all linear layers
        head_fn: can be torch.exp - for log space - might be good for Fokker-Planck pdes
        """
        super().__init__()
        self.head_fn = head_fn
        self.ff = ff
        self.rwf = rwf
        self.modified_mlp = modified_mlp
        self.act = activation_fn()
        in_dim = ff.output_dim if ff else input_dim

        if modified_mlp:
            self.enc1 = _linear(in_dim, layers[0], **rwf)
            self.enc2 = _linear(in_dim, layers[0], **rwf)
            self.hidden = nn.ModuleList()
            self.hidden.append(_linear(in_dim, layers[0], **rwf))
            for l1, l2 in zip(layers[:-1], layers[1:]):
                self.hidden.append(_linear(l1, l2, **rwf))
            self.out_layer = _linear(layers[-1], output_dim, **rwf)
        else:
            net_layers = []
            for l1, l2 in zip(layers[:-1], layers[1:]):
                net_layers.append(_linear(l1, l2, **rwf))
                net_layers.append(activation_fn())
            first = _linear(in_dim, layers[0], **rwf)
            self.net = nn.Sequential(
                first, activation_fn(),
                *net_layers,
                _linear(layers[-1], output_dim, **rwf)
            )

    def forward(self, X):
        with record_function("forward"):
            x = self.ff(X) if self.ff else X
            if self.modified_mlp:
                U = self.act(self.enc1(x))
                V = self.act(self.enc2(x))
                UmV = U - V
                h = x
                for layer in self.hidden:
                    z = self.act(layer(h))
                    h = V + z * UmV
                return self.head_fn(self.out_layer(h),X)
            else:
                return self.head_fn(self.net(x),X)


#d=5:  num_freqs=128, sigma=8, hidden_width=128, num_blocks=4
#d=10: num_freqs=256, sigma=15, hidden_width=256, num_blocks=5
#d=15: num_freqs=320, sigma=30, hidden_width=256, num_blocks=6
#d=20: num_freqs=512, sigma=40, hidden_width=256, num_blocks=8
class ResPINN(nn.Module):
    def __init__(self, D, hidden_width=256, num_blocks=5,
        ff=None,
        modified_mlp=False,
        rwf={}
        ):
        super().__init__()
        self.ff = ff
        self.modified_mlp = modified_mlp
        in_dim = ff.output_dim if ff else D

        self.first = _linear(in_dim, hidden_width, **rwf)
        self.act = nn.Mish()

        if modified_mlp:
            self.enc1 = _linear(in_dim, hidden_width, **rwf)
            self.enc2 = _linear(in_dim, hidden_width, **rwf)

        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_width, rwf=rwf)
            for _ in range(num_blocks)
        ])
        self.out = _linear(hidden_width, 1, **rwf)

    def forward(self, X):
        with record_function("forward"):
            x = self.ff(X) if self.ff else X
            h = self.act(self.first(x))

            if self.modified_mlp:
                U = self.act(self.enc1(x))
                V = self.act(self.enc2(x))
                UmV = U - V
                for block in self.blocks:
                    z = self.act(block(h))
                    h = V + z * UmV
            else:
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

