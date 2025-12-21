

Parabolic PDEs

- Heat equation

- Fokker Planck (Diffusion equation + drift)

- Schroedinger ??

- Black Scholes



# Second order parabolic


## Definition

Let
- $\Omega R^d$ be open and bounded
- $T \in (0, \infty)$

Set
- $Q_T = (0,T) \times \Omega$

Consider

$$
\partial_t u + L\,u = f\,, in\, Q_T \\
u = 0\,, on\, (0,T)\times \Omega \\
u(0,\cdot) = g\,, on\, \Omega
$$

Where
- $u: Q_T \to R$
- $f: Q_T \to R$
- $g: \Omega \to R$

### Notes

$$
L\,u := - \sum_{i,j=1}^d \frac{\partial}{\partial x_i}(a_{ij}\frac{\partial u}{\partial x_j}) + \sum_{i=1}^d b_i \frac{\partial u}{\partial x_i} + c\, u \\
= -\,div (A \, \nabla u) + \vec{b} \cdot \nabla u + c\, u
$$

Where 
- $ a_{ij},\, b_i,\, c \: : \, Q_T \to R $

Operator $\partial_t + L$ is parabolic if $\exists \theta>0$ such that
$$
\vec{\xi}^T A(t,x)\,\vec{\xi} \ge \theta ||\xi||^2
$$
for 
$ \forall \xi\in R_d $
and 
$ \forall(t,x)\in Q_T $


## Examples

### 1. Heat equation

Taking
$$
A=I_{d\times d},\:\: \vec{b} = 0,\:\: c = 0
$$

We get
$$
\partial_t u - \Delta u = f
$$

### 2. Fokker-Planck equation

Taking
$$
A=\alpha\,I_{d\times d},\:\: \vec{b} = \nabla w,\:\: c = \Delta w
$$

Where
- $\alpha \in R$
- $w : Q_T \to R$

We get
$$
\partial_t u - \alpha\,\Delta u + \nabla w \cdot \nabla u + \Delta w\: u = 0
$$

Using standard notation
- $ \alpha = D $
- $ u(t,x) = p(t,x) $
- $ w(t,x) = \mu\,v(t,x) $
$$
\partial_t p - D\,\Delta p + \mu\,\nabla v\cdot \nabla p + \mu\,\Delta v\: p = 0 \\
\partial_t p - D\,\Delta p + \mu\,\nabla \cdot (p\,\nabla v) = 0
$$

We get
$$
\partial_t p = D\,\Delta p - \mu\,\nabla \cdot (p\,\nabla v)
$$

Where
- $D,\, \mu \, \in R$
- $v : Q_T \to R$
















































