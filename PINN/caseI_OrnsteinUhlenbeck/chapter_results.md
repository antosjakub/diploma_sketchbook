
# Ornstein-Uhlenbeck process

We consider the Ornstein-Uhlenbeck (OU) process with anisotropic and correlated noises, 

$$
dX_t = -\frac{1}{2} x\,dt + \Sigma^\frac{1}{2} dW_t
$$

The Brownian noise is correlated with the covariance matrix $\Sigma\in\mathbb{R}^{d\times d}$, which is constructed as follows:
- We generate random orthogonal matrix $Q$ from QR decomposition, serving as the eigenspace of $\Sigma$.
- $\Sigma$’s eigenvalues $\Gamma = diag(\lambda_1, \lambda_2, · · · , \lambda_d)$ where $\lambda_{2i} ∼ Uniform([1, 1.1])$ and $\lambda_{2i+1} = 1/\lambda_{2i}$.
- Finally, $\Sigma = Q^T \Gamma Q$

We have
$$
\mu = -\frac{1}{2} x \\ 
\sigma = \Sigma^\frac{1}{2}
$$
Thus the corresponding PDE then reads
$$
\partial_t p - \frac{1}{2} \nabla\cdot (\Sigma\nabla p + x\,p) = 0\\
\partial_t p + \nabla\cdot J = 0\\
$$
bc:
$$
J\cdot n = 0\\
(\Sigma\nabla p + x\,p)\cdot n = 0
$$



We consider 3 distribution for the initial condition:

- gaussian (has analytic solution)
- cauchy (no analytic solution)
- laplace (no analytic solution)

show the plots - density & vector field for each distrib



## Case 1: p_0(x) = unit gaussian

$$
p_0(x) = \alpha_d\, e^{-\frac{1}{2}||x||^2}\:, \quad \alpha_d = (2\pi)^{-d/2}
$$

$$
q_0(x) = \log p_0(x) = \log \alpha_d - \tfrac{1}{2}||x||^2
$$

$$
s_0(x) = \nabla_x \log p_0(x) = -x
$$

Here, the SDE solution is anisotropic, which is a Gaussian $p_t(x) ∼ N(0, \Sigma_t)$,
where $\Sigma_t = e^{−t}I + (1 − e^{−t})\Sigma$.
The exact score function is $s_t(x) = \Sigma_t^{-1} x$.
The SDE has finite variance and gradually transforms the unit Gaussian to $N (0, \Sigma)$ as $t \to \infty$.

### Score-PINN



## Case 2: p_0(x) = cauchy distrib

$$
p_0(x) = \alpha_d\, \frac{1}{(1+||x||^2)^\frac{d+1}{2}}\:, \quad \alpha_d = \frac{\Gamma\!\left(\tfrac{d+1}{2}\right)}{\pi^{(d+1)/2}}
$$

$$
q_0(x) = \log p_0(x) = \log \alpha_d - \tfrac{d+1}{2}\,\log\!\left(1 + ||x||^2\right)
$$

$$
s_0(x) = \nabla_x \log p_0(x) = -(d+1)\frac{x}{1 + ||x||^2}
$$


## Case 3: p_0(x) = laplace distrib

$$
p_0(x) = \alpha_d\,  \prod_i e^{-|x_i|}\:, \quad \alpha_d = 2^{-d}
$$


### Vanilla PINN


### Score PINN

We get

$$
q_0(x) = \log p_0(x) = \log \alpha_d - \sum_i |x_i|
$$

$$
s_0(x) = \nabla_x \log p_0(x) = -\operatorname{sign}(x)
$$

where $\operatorname{sign}(x)$ for $x\in\R^d$ is acting component-wise, that is
$\operatorname{sign}(x)|_i := \operatorname{sign}(x_i)$

note: p,q,s discontinuous 


We consider the following variants:

- original score pinn formulation
    - with hardcoded bc ic
    - trajs sampling
- variants
    - full domain - not only trajs
    - with loss in bc ic
    - with loss in bc
    - 1k per epoch
    - looping batches