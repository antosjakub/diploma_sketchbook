

Parabolic PDEs

- Heat equation

- Fokker Planck (Diffusion equation + drift)

- Schroedinger ??

- Black Scholes



# Second order parabolic


## Definition

Let
- $\Omega \subset R^d$ be open and bounded
- $T \in (0, \infty)$

Set
- $Q_T = (0,T) \times \Omega$

Consider

$$
\partial_t u + L\,u = f\quad \text{in } Q_T \\
u(t,x) = 0\quad \text{on } (0,T)\times \Omega \\
u(0,x) = g(x)\quad \text{on } \Omega
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
$$
\partial_t u - \alpha\,\Delta u + \nabla\cdot(u~\nabla w) = 0
$$

Using standard notation
- $ \alpha = D $
- $ u(t,x) = p(t,x) $
- $ w(t,x) = \mu\,v(t,x) $
$$
\partial_t p - D\,\Delta p + \mu\,\nabla \cdot (p\,\nabla v) = 0
$$

We get
$$
\partial_t p = D\,\Delta p - \mu\,\nabla \cdot (p\,\nabla v)
$$

Where
- $D,\, \mu \, \in R$
- $v : Q_T \to R$





# Application: evolution of a molecule with a LJ potential


Evolution of a molecule with 7 atoms = PDE in 21 dimensions.

We suppose that the inertia / velocity of the atoms is small, hence we do not get dependence on velocity - only 21 and not 42 dim state space.

Here, we collect the spatial coordinates of the $n$ individual atoms into a single coordinate vector, $x\in\mathbb{R}^{d}$, where $d=3n$.
Instead of $x$, we will be sometimes using $(x_1,..,x_d)$, $(x_1,..,x_{3n})$, or $(\vec{x}_1,...,\vec{x}_{n})$, where $\vec{x}_i$ contains the 3 spatial coordinates of the $i$-th atom.

Compare against results giving the optimal time of transition from state $x_0$ to $x_f$ via paths in configuraiton space.

## From langevin to Fokker-Planck

Consider a Langevin equation describing our system
$$
d\mathbf{x} = -\frac{\nabla U(\mathbf{x})}{\zeta}\,dt + \sqrt{2D}\,d\mathbf{W}
$$

In general, the Langevin equation reads:
$$
dX_t = \mu(X_t,t)\,dt + \sigma(X_t,t)\,dW_t, \quad D(X_t,t) = \frac{\sigma^2(X_t,t)}{2}
$$

The corresponding PDE is the Fokker–Planck equation for the probability density $p(\mathbf{x},t)$

$$
\frac{\partial \,p(x,t)}{\partial t} = - \nabla \cdot [\, \vec{\mu}(x,t)\,p(x,t) \,] + D\,\Delta\,p(x,t) 
$$

We have
$$
\mu = -\frac{\nabla U}{\zeta}, \quad\sigma = \sqrt{2D}
$$

Thus

$$
\frac{\partial p(x,t)}{\partial t} = \nabla \cdot [\,\frac{\nabla U(x)}{\zeta}\,p(x,t)\,] + D\,\Delta\,p(x,t)
$$
$$
\frac{\partial p(x,t)}{\partial t} = \frac{\Delta U(x)}{\zeta} \,p(x,t) + \frac{\nabla U(x)}{\zeta}\cdot\nabla p(x,t) + D\,\Delta\,p(x,t)
$$
In a more compact notation:
$$
\partial_t p = \frac{\Delta U}{\zeta} \,p + \frac{\nabla U}{\zeta}\cdot\nabla p + D\,\Delta\,p
$$
We need to compute:
$$
\partial_t p\,, ~\nabla p\,, ~\Delta p \\
\nabla U\,, ~\Delta U
$$

Where $U(x)$ is given as:
$$
U(x) = U(x_1,..,x_{3n}) = U(\vec{x}_1,...,\vec{x}_{n}) = \sum_{i=1,~j\ge i+1}^n u(\|\vec{x}_i-\vec{x}_j\|)
$$
$$
u(r) = 4\epsilon~ \Big[\,\Big(\frac{r_0}{r}\Big)^{12}-\Big(\frac{r_0}{r}\Big)^6\,\Big]
$$



