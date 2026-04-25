
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
\partial_t p = \frac{1}{2} div(\Sigma\nabla p) + \frac{1}{2}\nabla\cdot(x\,p)\\
\partial_t p = \frac{1}{2} div(\Sigma\nabla p) + \frac{1}{2}dp + \frac{1}{2}x\cdot \nabla p \\
$$

$$
\partial_t p - \frac{1}{2} \nabla\cdot (\Sigma\nabla p + x\,p) = 0\\
\partial_t p + \nabla\cdot J = 0\\
$$
bc:
$$
J\cdot n = 0\\
(\Sigma\nabla p + x\,p)\cdot n = 0
$$

To get rid of the mixed second derivatived caused by the $\Sigma$ matrix, we consider the change of coordinates
$y = A\,x$
where A is a general matrix whose form is to be determined.

The change of coordinates introduces a jacobian terms into the PDE, namely

$$
\frac{\partial p}{\partial x_i}
= \frac{\partial p}{\partial y_j}\frac{\partial y_j}{\partial x_i}
= \frac{\partial p}{\partial y_j}\frac{\partial}{\partial x_i}
\left( A_{jk}x_k \right)
= \frac{\partial p}{\partial y_j}A_{ji}
$$

Hence
$$
\nabla_x p = A^T\nabla_y p
$$

Now consider a general vector function $\vec{v}(x):\R^d\to\R^d$

$$
div_x(\vec{v})
= \frac{\partial v_i}{\partial x_i}
= ...
= \frac{\partial v_i}{\partial y_j}A_{ji}
= \frac{\partial}{\partial y_j}(A_{ji}v_i)
= div_y(A\vec{v})
$$

Setting $\vec{v} = \Sigma\nabla_x p$ and combining the previous two results, we obtain:

$$
div_x(\Sigma\nabla_x p)
= div_y(A\Sigma A^T\nabla_y p)
$$

The PDE thus transform from

$$
\partial_t p
= \frac{1}{2} div_x(\Sigma\nabla_x p)
+ \frac{1}{2}x\cdot \nabla_x p
+ \frac{1}{2}d\,p
$$

to
$$
\partial_t p
= \frac{1}{2} div_y(A\Sigma A^T\nabla_y p)
+ \frac{1}{2}(A^{-1}y)\cdot (A^T\nabla_y p)
+ \frac{1}{2}d\,p
$$

And since
$$
(A^{-1}y)\cdot (A^T\nabla_y p)
= (A^{-1}y)^T (A^T\nabla_y p)
= y^TA^{-T}A^T\nabla_y p
= y^T\nabla_y p
$$

The system further simplifiesa to
$$
\partial_t p
= div_y(A\Sigma A^T\nabla_y p)
+ \frac{1}{2}y\cdot \nabla_y p
+ \frac{1}{2}d\,p
$$

Now, we know that $\Sigma=Q^T\Gamma Q$, implying
$$
A\Sigma A^T
= A Q^T \Gamma Q A^T
$$
Setting $A=\Gamma^{-1/2}Q$, we obtain $A\Sigma A^T = I$.

Lastly, setting $p(x,t)=p(A^{-1}y,t)=:\tilde{p}(y,t)$

The PDE system thus becomes
$$
\partial_t \tilde{p}
= \Delta_y \tilde{p}
+ \frac{1}{2}y\cdot \nabla_y \tilde{p}
+ \frac{1}{2}d\,\tilde{p}
\\
\tilde{p}(y,0) = p_0(Q^T\Gamma^{1/2}\,y)
$$

We then reconstruct the solution as $p(x,t) = \tilde{p}(\Gamma^{-1/2}Q\,x,t)$



We consider 3 distribution for the initial condition:

- gaussian (has analytic solution)
- cauchy (no analytic solution)
- laplace (no analytic solution)


The pde can be simplified by considering the change of variables:

$$
y = \Gamma^{-1/2}Q x
$$


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

