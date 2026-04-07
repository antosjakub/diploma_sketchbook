# The Fokker-Planck equation

$$
\partial_t\,p - D\,\Delta p
- \frac{1}{\zeta}\nabla p \cdot \nabla U - \frac{1}{\zeta}p\,\Delta U = 0
$$

Consider a box of size of $L$, that is $\Omega=[0,L]^d$ and $t\in[0,T]$.

We want to normalize the domain to $\hat{\Omega}=[0,1]^d,\:\:\hat{t}\in[0,1]$

We consider the transformation

$$
x=L\hat{x} \\
t=T\hat{t}
$$

$$
\hat{U}(\hat{x}) := U(L\hat{x}) = U(x) \\
\hat{p}(\hat{x}, \hat{t}) := p(L\hat{x}, T\hat{t}) = p(x, t)
$$

The derivates transform as

$$
\frac{\partial}{\partial x_i} = \frac{1}{L} \frac{\partial}{\partial \hat{x}_i} \\
\frac{\partial}{\partial t} = \frac{1}{T} \frac{\partial}{\partial \hat{t}_i}
$$

We thus get:

$$
\partial_t\,p(x,t) - D\,\Delta p(x,t)
- \frac{1}{\zeta}\nabla p(x,t) \cdot \nabla U(x)
- \frac{1}{\zeta}p(x,t)\,\Delta U(x) = 0
$$

$$
\partial_t\,p(L\hat{x},T\hat{t})
- D\,\Delta p(L\hat{x},T\hat{t})
- \frac{1}{\zeta}\nabla p(L\hat{x},T\hat{t}) \cdot \nabla U(L\hat{x})
- \frac{1}{\zeta}p(L\hat{x},T\hat{t})\,\Delta U(L\hat{x}) = 0
$$



$$
\frac{1}{T}\hat{\partial_t}\,\hat{p}(\hat{x},\hat{t})
- \frac{1}{L^2}\,\hat{\Delta} \hat{p}(\hat{x},\hat{t})
- \frac{1}{L^2}\,\frac{1}{\zeta}\hat{\nabla} \hat{p}(\hat{x},\hat{t}) \cdot \hat{\nabla} \hat{U}(\hat{x})
- \frac{1}{L^2}\,\frac{1}{\zeta}\hat{p}(\hat{x},\hat{t})\,\hat{\Delta} \hat{U}(\hat{x}) = 0
$$


$$
\hat{\partial_t}\,\hat{p}(\hat{x},\hat{t})
- \frac{T}{L^2}\, \Bigg( \hat{\Delta} \hat{p}(\hat{x},\hat{t})
+ \frac{1}{\zeta}\hat{\nabla} \hat{p}(\hat{x},\hat{t}) \cdot \hat{\nabla} \hat{U}(\hat{x})
+ \frac{1}{\zeta}\hat{p}(\hat{x},\hat{t})\,\hat{\Delta} \hat{U}(\hat{x}) \Bigg) = 0
$$


$$
\hat{\partial_t}\,\hat{p}
- \frac{T}{L^2}\, \Bigg( \hat{\Delta} \hat{p}
+ \frac{1}{\zeta}\hat{\nabla} \hat{p} \cdot \hat{\nabla} \hat{U}
+ \frac{1}{\zeta}\hat{p}\,\hat{\Delta} \hat{U} \Bigg) = 0
$$



## PINNs

### sampling

x corresponding to 2 atoms close to each other -> U and hence also U_grad, U_laplace explodes -> inf loss

solution: filter out such points


### train IC first

- the potentials make the problem hard to train - stiff?
- 1st train only IC, save model, then train full
- full train: start with t=0, slowly progress to t=T
- clip gradients


### architecture

- exp head - easier to reach low values 1e-10 etc.





N1

sample IC, filter evil ones
loss for IC only
custom err metrics
