from torch import nn
import torch
import sys

t = sys.argv[1]
print(40*"==")
print(40*"==")
print("t =", t)
t = int(t)


#dtype = torch.float32
torch.manual_seed(42)

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
model = PINN(d, 5*[64], 1)

#compile = True
#if compile:
model.compile()


bs = 100
X = torch.rand((bs,d))
print(X[0,:])



import derivatives
from torch.func import jacrev, jacfwd, jvp, vjp, grad, vmap, hessian

#t = 3
#print("t =", t)
if t==0:
    u, grad_u = derivatives.compute_u_grad_u(model, X)
    print(grad_u[0])
elif t==1:
    X.requires_grad_(True)
    u = model(X)
    grad_u = derivatives.compute_grad(X, u, torch.ones_like(u))
    print(grad_u[0])
elif t==2:
    # not working with .compile
    X.requires_grad_(True)
    u, grad_u, lapl_u = derivatives.compute_derivatives(model, X)
    lapl = lapl_u.sum(dim=-1)
    print(lapl[:5])

elif t==3:
    # throws error
    u = model(X)
    grad_u = vmap(grad(model))(X)
    print(grad_u[0])
elif t==4:
    u = model(X)
    grad_u = vmap(jacrev(model))(X)
    print(grad_u[0])

elif t==5:
    H = vmap(hessian(model))(X)
    lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    print(lapl[:5,0])
    #print(H.shape)
    #print(lapl.shape)
elif t==6:
    H_fn = jacfwd(jacrev(model))
    H = vmap(H_fn)(X)
    lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    print(lapl[:5,0])
elif t==7:
    H_fn = jacrev(jacrev(model))
    H = vmap(H_fn)(X)
    lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    print(lapl[:5,0])

else:


    def train(x):
        u = model(x)

        grad_u = vmap(jacrev(model))(X)

        H = vmap(hessian(model))(X)
        lapl = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        loss = torch.mean(
            u + grad_u.sum(dim=-1) + lapl
        )

        loss = torch.mean(u)
        loss.backward()

    train_com = torch.compile(train)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_com(X)
    optimizer.step()
