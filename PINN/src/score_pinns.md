
what chages:
- training 1
    - Score PDE
    - loss - sampled over trajectories
    - different pde system to optimize
- training 2
    - LL-ODE
    - loss normal?



sampling not preformed on the whole grid - only on the regions with high prob density



$$
dX_t = \mu(X_t,t) dt + \sigma(X_t,t) dX_t
$$

$$
\partial_t p
= \frac{1}{2} \sum_{i,j=1}^d \frac{\partial^2}{\partial x_i\,\partial x_j}
\Bigg[\sum_{k=1}^d \sigma_{ik}(x,t)\sigma_{jk}(x,t)\,p(x,t) \Bigg]
- \sum_{i=1}^d  \frac{\partial}{\partial x_i} \Big[ \mu_i(x,t)\,p(x,t) \Big]
$$

$$
\partial_t p
= \sum_{i,j=1}^d \frac{\partial^2}{\partial x_i\,\partial x_j}
\Big[D_{ij}(x,t)\,p(x,t) \Big]
- \big(\nabla\cdot \mu(x,t)\big)\,p(x,t) - \mu(x,t)\cdot \nabla p(x,t)
$$



LL-ODE

$$
\partial_t q
= \mathcal{L}(s)
= \frac{1}{2}\nabla\cdot(\sigma\sigma^Ts)
+ \frac{1}{2}||\sigma^T s||^2
- A\cdot s - \nabla\cdot A \\
q(x,0) = \log p_0(x)
$$
where
$$
A(x,t) := \mu(x,t) - \frac{1}{2} \nabla \cdot \Big[ \sigma(x,t)\sigma(x,t)^T \Big]
$$
where the divergence of the matrix ($\nabla \cdot \Big[ \sigma(x,t)\sigma(x,t)^T \Big]$) is taken row wise, that is
$$
(\nabla \cdot M)_i = \sum_{j=0}^{d} \frac{\partial M_{ij}}{\partial x_j}
$$

the Score PDE
$$
\partial_t s = \nabla\mathcal{L}(s) \\
s(x,0) = \nabla \log p_0(x)
$$


log likelihood
$$
q_t(x) = \log(x)
$$

score
$$
s_t(x) = \nabla\log(x)
$$


loss




---
consider

$$
\mu(x,t) = \mu(x) \\
\sigma(x,t) = \sigma\,I
$$

LL-ODE

$$
\partial_t q
= \frac{\sigma^2}{2}\nabla\cdot s
+ \frac{\sigma^2}{2}||s||^2
- \mu\cdot s - \nabla\cdot \mu \\
q(x,0) = \log p_0(x)
$$

then Score PDE

$$
\partial_t s
= \frac{\sigma^2}{2}\Delta s
+ \frac{\sigma^2}{2}\nabla||s||^2
- \nabla(\mu \cdot s)
- \Delta \mu \\
s(x,0) = \nabla\log p_0(x)
$$

---
Isotripic OU process

$$
dX_t = \mu(X_t,t) dt + \sigma(X_t,t) dX_t
$$

$$
\mu(x,t) = -\frac{1}{2}x \\
\sigma(x,t) = \sigma I
$$

LL ODE:
$$
\partial_t q = \frac{\sigma^2}{2} (\nabla\cdot s + ||s||^2) + \frac{1}{2}( x \cdot s + d)
$$

Score PDE:
$$
\partial_t s = \nabla\cdot \Bigg[
    \frac{\sigma^2}{2} (\nabla\cdot s + ||s||^2) + \frac{1}{2}( x \cdot s + d)
\Bigg]
$$




---

preparation

- $$ s_0 = \nabla \log p_0 $$
- score pde system

loss calculation:
sample to get $x_0\sim p_0$ (use to compute ic loss)
then obtain samples $x\sim p$ using SDE discretization

then:
choose random time index and random trajectory index, get 1000 such pairs - this is your X

