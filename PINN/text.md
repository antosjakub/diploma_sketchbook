


u_t = alpha u_laplace

ai \in (0,4pi)

u = sin(a1 x1) ... sin(an xn) * e^(-alpha*() t)
u_t = -alpha*() u
u_laplace = -(a1^2 + ... + an^2) u


decay by 0.9 every 2000 epochs

lr_init = 0.001

lr_init 0.9 = lr_init * ?**2000



compute spatial gradents
compute pde residuum
calc loss
grad descent step


d-dim input
2nd der problematic
- not so if we only have second der of the same coordinate -> d ders





# PINNS



## 



## why it might work

automatic differentiation - no numerical, but symbolic graph - also a challange

motivation behind ML / deep learning / PINNS:
(wikipedia: https://en.wikipedia.org/wiki/Physics-informed_neural_networks)
Recently, solving the governing partial differential equations of physical phenomena using deep learning has emerged as a new field of scientific machine learning (SciML), leveraging the universal approximation theorem[4] and high expressivity of neural networks. In general, deep neural networks could approximate any high-dimensional function given that sufficient training data are supplied.
[4] = https://www.sciencedirect.com/science/article/abs/pii/0893608089900208

## architecture

MPL

loss = pde residual

random sampling (vs mini-batch)


## specific network choice

tanh activations


## crutial in high dim

sampling strategies
parallelization

