from torch import nn
import torch

import time
import utility

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--compile_model", default="0", type=str)
parser.add_argument("--train_type", default="train_h", type=str, help="")
args = parser.parse_args()
args.compile_model = int(args.compile_model)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)
print(f"- device = {device}")

class PINN(nn.Module):
    def __init__(self, input_dim, layers=[64], output_dim=1, activn_fn=nn.Tanh):
        super().__init__()

        net_layers = []
        for l1, l2 in zip(layers[:-1], layers[1:]):
            net_layers.append(nn.Linear(l1, l2))
            net_layers.append(activn_fn())
        first = nn.Linear(input_dim, layers[0])
        self.net = nn.Sequential(
            first, activn_fn(),
            *net_layers,
            nn.Linear(layers[-1], output_dim)
        )

    def forward(self, X):
        return self.net(X)


d = 6
model = PINN(d, 3*[512], 1).to(device)

if args.compile_model:
    print("- compile_model = true")
    model.compile()
    #model = torch.compile(model, mode="reduce-overhead")
    #model = torch.compile(model, mode="max-autotune")
else:
    print("- compile_model = false")


from torch.func import jacrev, jacfwd, jvp, vjp, grad, vmap, hessian

if args.train_type == "train_h":
    H_fn = hessian(model)
elif args.train_type == "train_rr":
    H_fn = jacrev(jacrev(model))
elif args.train_type == "train_ff":
    H_fn = jacfwd(jacrev(model))
elif args.train_type == "train_fr":
    H_fn = jacrev(jacfwd(model))
elif args.train_type == "train_rf":
    H_fn = jacfwd(jacfwd(model))

def train_h(x):
    H = vmap(H_fn)(x)
    lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    loss = torch.mean(
        lapl
    )
    loss.backward()

def train_rr(x):
    H = vmap(H_fn)(x)
    lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    loss = torch.mean(
        lapl
    )
    loss.backward()

def train_fr(x):
    H = vmap(H_fn)(x)
    lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    loss = torch.mean(
        lapl
    )
    loss.backward()

def train_rf(x):
    H = vmap(H_fn)(x)
    lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    loss = torch.mean(
        lapl
    )
    loss.backward()

def train_ff(x):
    H = vmap(H_fn)(x)
    lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    loss = torch.mean(
        lapl
    )
    loss.backward()

import derivatives
def train_jvp(x):
    n_spatial = x.shape[1] - 1
    basis = torch.eye(n_spatial, device=x.device, dtype=x.dtype)

    def scalar_model(xi):
        return model(xi.unsqueeze(0)).squeeze()

    grad_fn = grad(scalar_model)

    def lapl_point(xi):
        def second_along(v):
            tangent = torch.zeros_like(xi)
            tangent[:n_spatial] = v
            _, hvp = jvp(grad_fn, (xi,), (tangent,))
            return hvp[:n_spatial] @ v

        return vmap(second_along)(basis).sum()

    lapl = vmap(lapl_point)(x)
    loss = torch.mean(
        lapl
    )
    loss.backward()

def train_class(x):
    x.requires_grad_(True)
    u, grad_u, lapl_u = derivatives.compute_derivatives(model, x)
    lapl = lapl_u.sum(dim=-1)
    loss = torch.mean(
        lapl
    )
    loss.backward()


if args.train_type == "train_h":
    train = train_h
elif args.train_type == "train_rr":
    train = train_rr
elif args.train_type == "train_ff":
    train = train_ff
elif args.train_type == "train_fr":
    train = train_fr
elif args.train_type == "train_rf":
    train = train_rf
elif args.train_type == "train_jvp":
    train = train_jvp
elif args.train_type == "train_class":
    train = train_class
print("- train_type =", train.__name__)


n_steps = 10
n_warmup = 5
bs = 500
N = bs * (n_warmup + n_steps)
X = torch.rand((N, d), device=device)

import  os
os.makedirs('compile_runs', exist_ok=True)

#prof = utility.Profiler(f"compile_runs/compile_rep_{args.train_type}", 0, n_steps+n_warmup-1)
#prof.make()
#prof.start(0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ts = []
t_avrg = 0.0
j = 0
for i in range(n_warmup+n_steps):
    if i > n_warmup:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    train(X[bs*i:bs*(i+1),:])
    optimizer.step()
    if i > n_warmup:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        t_delta = t2-t1
        ts.append(t_delta)
        t_avrg += t_delta
        #t_delta = utility.get_duration_h_m_s(t1, t2, "")

nt = len(ts)
t = t_avrg / nt
print(f"t_avrg = {t:.6f}")
print("ts = [", ", ".join([f"{x:.5f}" for x in ts]) + " ]")
#prof.exit(n_steps+n_warmup-1)
