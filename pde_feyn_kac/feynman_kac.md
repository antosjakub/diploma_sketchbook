# The Feynman-Kac Formula: A Bridge Between PDEs and Stochastic Processes

---

## Chapter 1. Introduction — Why Probability Enters the Theory of PDEs

### 1.1 Motivation: the curse of dimensionality

Classical numerical methods for partial differential equations — finite differences, finite elements, spectral methods — discretise the spatial domain on a grid. In $d$ dimensions with $N$ points per coordinate axis, the resulting grid contains $N^d$ degrees of freedom. For $d = 3$ and $N = 100$ this is $10^6$, which is routine. For $d = 21$ (the dimension of the configuration space of a molecule with 7 atoms, each having 3 spatial coordinates) even $N = 10$ yields $10^{21}$ grid points — a number that exceeds the capacity of any foreseeable computer.

The Feynman-Kac formula offers a fundamentally different route. It expresses the solution $u(t,x)$ of a parabolic PDE at a *single point* $(t,x)$ as the *expected value* of a random variable constructed from sample paths of a stochastic differential equation. This expectation can be approximated by Monte Carlo sampling, whose convergence rate $\mathcal{O}(N^{-1/2})$ is **independent of the spatial dimension** $d$. For high-dimensional problems, this independence is the decisive advantage.

### 1.2 Historical context

The formula bears the names of two physicists who arrived at the same mathematical structure from opposite directions. In the late 1940s Richard Feynman developed his path-integral formulation of quantum mechanics, expressing transition amplitudes as formal "sums over all histories" of a particle. These integrals, taken over a space of continuous paths, were physically compelling but lacked rigorous mathematical justification.

Mark Kac, a mathematician at Cornell, attended a seminar by Feynman in 1947 and recognised that by replacing the imaginary unit $i$ of quantum mechanics with $-1$ (a "Wick rotation" from the Schrödinger equation to the heat equation) one could make the path integral rigorous using Norbert Wiener's theory of Brownian motion. Kac published the result in his 1949 paper *"On Distributions of Certain Wiener Functionals"* [1]. The complex-valued version needed for quantum mechanics proper remains an open problem.

### 1.3 The core idea, informally

Consider the heat equation $\partial_t u = \tfrac{1}{2}\Delta u$ in $\mathbb{R}^d$ with initial condition $u(0,x) = g(x)$. Its solution is the convolution of $g$ with the Gaussian heat kernel:

$$
u(t,x) = \int_{\mathbb{R}^d} g(y)\,\frac{1}{(2\pi t)^{d/2}}\,e^{-|x-y|^2/(2t)}\,dy.
$$

But the Gaussian kernel is nothing other than the transition density of a $d$-dimensional Brownian motion $W_t$. Hence the integral can be rewritten as

$$
u(t,x) = \mathbb{E}\!\left[\,g(x + W_t)\,\right].
$$

The solution of the heat equation at the point $(t,x)$ equals the mean value of $g$ evaluated at the random position reached by a Brownian motion that starts at $x$ and runs for time $t$. The Feynman-Kac formula generalises this observation to a wide class of parabolic PDEs — those with drift, potential (zeroth-order) terms, and source terms — by associating to each PDE an appropriate stochastic differential equation and an appropriate weighting of its paths.

### 1.4 Outline

The remainder of this text is organised as follows.

- **Chapter 2** develops, from scratch, the minimal stochastic calculus needed: Brownian motion, the Itô integral, Itô's lemma, and the notion of a stochastic differential equation. The presentation is tailored to readers who are fluent in PDEs but have not encountered stochastic processes.
- **Chapter 3** states and proves the Feynman-Kac formula, discusses its variants, and applies it to concrete PDE problems — including several from our problem set of high-dimensional diffusion equations.


---

## Chapter 2. Stochastic Calculus for PDE Specialists

This chapter introduces the stochastic machinery that underpins the Feynman-Kac formula. We develop only what is strictly needed: Brownian motion, the Itô integral, Itô's lemma, and stochastic differential equations (SDEs). Proofs that would require measure-theoretic probability are stated precisely but not proved in full; we give references instead.

Throughout, $(\Omega, \mathcal{F}, \mathbb{P})$ denotes a probability space equipped with a filtration $(\mathcal{F}_t)_{t \ge 0}$ satisfying the usual conditions (right-continuity and completeness).

### 2.1 Brownian motion

**Definition 2.1 (Standard Brownian motion).** A stochastic process $W = (W_t)_{t \ge 0}$ taking values in $\mathbb{R}^d$ is called a *standard $d$-dimensional Brownian motion* (or Wiener process) if

1. $W_0 = 0$ almost surely.
2. $W$ has *independent increments*: for $0 \le t_1 < t_2 < \cdots < t_n$, the random vectors $W_{t_2}-W_{t_1},\; W_{t_3}-W_{t_2},\;\dots,\;W_{t_n}-W_{t_{n-1}}$ are independent.
3. $W$ has *stationary Gaussian increments*: $W_t - W_s \sim \mathcal{N}(0,(t-s)\,I_d)$ for every $0 \le s < t$.
4. $t \mapsto W_t(\omega)$ is continuous for almost every $\omega \in \Omega$.

**Key property for calculus: quadratic variation.** A classical smooth path $f\colon [0,T]\to\mathbb{R}$ has zero quadratic variation: $\sum |f(t_{k+1})-f(t_k)|^2 \to 0$ as the partition is refined. A Brownian path does not. For a one-dimensional Brownian motion, the quadratic variation over $[0,t]$ satisfies

$$
[W,W]_t \;:=\; \lim_{|\Pi|\to 0}\sum_{k} (W_{t_{k+1}} - W_{t_k})^2 \;=\; t \qquad\text{(limit in probability)}.
$$

It is this property that makes stochastic calculus differ from ordinary calculus. Informally, one writes the mnemonic rule

$$
(dW_t)^2 = dt, \qquad (dt)^2 = 0, \qquad dW_t\,dt = 0.
$$

These rules will be made precise by Itô's lemma.

### 2.2 The Itô integral

To define $\int_0^T H_s\,dW_s$ for a suitable process $H$, one cannot use Riemann-Stieltjes integration because Brownian paths have unbounded variation on every interval. The Itô integral is instead constructed by a limiting procedure analogous to the Riemann integral but adapted to the irregularity of Brownian motion.

**Construction (sketch).** For *simple* (piecewise-constant, adapted) processes $H_s = \sum_k H_{t_k}\,\mathbf{1}_{(t_k,t_{k+1}]}(s)$, one defines

$$
\int_0^T H_s\,dW_s := \sum_k H_{t_k}\,(W_{t_{k+1}} - W_{t_k}).
$$

Note that $H$ is evaluated at the *left* endpoint $t_k$ of each subinterval — this is the defining choice of the Itô integral and the reason it has different algebraic properties from the Stratonovich integral. One then shows that this map $H \mapsto \int H\,dW$ is an isometry:

$$
\mathbb{E}\!\left[\left(\int_0^T H_s\,dW_s\right)^{\!2}\right] = \mathbb{E}\!\left[\int_0^T H_s^2\,ds\right] \qquad\text{(Itô isometry)},
$$

and extends by continuity to the closure of simple processes in $L^2(\Omega\times[0,T])$.

**Fundamental property.** If $\mathbb{E}\!\left[\int_0^T H_s^2\,ds\right] < \infty$, then the Itô integral $M_t := \int_0^t H_s\,dW_s$ is a *martingale*: for $s < t$,

$$
\mathbb{E}[M_t \mid \mathcal{F}_s] = M_s.
$$

In particular, $\mathbb{E}[M_t] = 0$. This martingale property is the engine of the Feynman-Kac proof: it is why stochastic integral terms vanish when one takes expectations.

### 2.3 Itô's lemma

**Theorem 2.2 (Itô's lemma).** Let $X_t \in \mathbb{R}^d$ be an Itô process of the form

$$
dX_t = \mu(t,X_t)\,dt + \sigma(t,X_t)\,dW_t,
$$

where $\mu\colon [0,T]\times\mathbb{R}^d \to \mathbb{R}^d$ and $\sigma\colon [0,T]\times\mathbb{R}^d \to \mathbb{R}^{d\times m}$, and let $f \in C^{1,2}([0,T]\times\mathbb{R}^d)$. Then

$$
df(t,X_t) = \left[\partial_t f + \mu \cdot \nabla_x f + \tfrac{1}{2}\operatorname{Tr}\!\left(\sigma\sigma^\top \operatorname{Hess}_x f\right)\right]dt + (\nabla_x f)^\top \sigma\,dW_t.
$$

In one dimension ($d=m=1$) with $dX_t = \mu\,dt + \sigma\,dW_t$, this reads

$$
df = \left(\partial_t f + \mu\,\partial_x f + \tfrac{1}{2}\sigma^2\partial_{xx}f\right)dt + \sigma\,\partial_x f\,dW_t.
$$

*Proof sketch.* Expand $f(t+dt, X_t+dX_t)$ to second order in $dX_t$ and first order in $dt$. The crucial point is that the second-order term $\tfrac{1}{2}\partial_{x_ix_j}^2 f\,dX^i\,dX^j$ does **not** vanish: since $dW^i\,dW^j = \delta_{ij}\,dt$, the product $dX^i\,dX^j$ has a nonzero $dt$-component equal to $(\sigma\sigma^\top)_{ij}\,dt$. All terms of order $(dt)^{3/2}$ or higher vanish in the limit. $\square$

**Comparison with the classical chain rule.** In ordinary calculus, the second-order term in a Taylor expansion is $\mathcal{O}(dt^2)$ and drops out. In Itô calculus it contributes at order $dt$ due to the quadratic variation of Brownian motion. This is the single most important distinction: *Itô's lemma is the chain rule plus a correction term that arises from the roughness of Brownian paths.*

### 2.4 Stochastic differential equations

**Definition 2.3.** A *stochastic differential equation* (SDE) in $\mathbb{R}^d$ is an equation of the form

$$
dX_t = \mu(t,X_t)\,dt + \sigma(t,X_t)\,dW_t, \qquad X_{t_0} = x_0,
$$

where $W_t$ is an $m$-dimensional Brownian motion, $\mu\colon [0,T]\times\mathbb{R}^d\to\mathbb{R}^d$ is the *drift coefficient*, and $\sigma\colon [0,T]\times\mathbb{R}^d\to\mathbb{R}^{d\times m}$ is the *diffusion coefficient*. The equation is shorthand for the integral equation

$$
X_t = x_0 + \int_{t_0}^t \mu(s,X_s)\,ds + \int_{t_0}^t \sigma(s,X_s)\,dW_s.
$$

**Theorem 2.4 (Existence and uniqueness; Itô, 1946).** If $\mu$ and $\sigma$ satisfy:

1. *Lipschitz continuity:* $|\mu(t,x)-\mu(t,y)| + |\sigma(t,x)-\sigma(t,y)| \le K|x-y|$,
2. *Linear growth:* $|\mu(t,x)| + |\sigma(t,x)| \le K(1+|x|)$,

for some constant $K$ and all $t \in [0,T]$, $x,y\in\mathbb{R}^d$, then for every initial condition $X_{t_0}=x_0$ there exists a unique *strong solution* adapted to the filtration generated by $W$.

*Proof.* By Picard iteration in the Banach space $L^2(\Omega; C([t_0,T];\mathbb{R}^d))$, using the Itô isometry at each step. See Øksendal [2, Theorem 5.2.1].

### 2.5 The infinitesimal generator and its connection to PDEs

Given the SDE $dX_t = \mu(t,X_t)\,dt + \sigma(t,X_t)\,dW_t$, define the second-order differential operator

$$
\mathcal{A} \;=\; \sum_{i=1}^d \mu_i(t,x)\,\partial_{x_i} + \frac{1}{2}\sum_{i,j=1}^d \left(\sigma\sigma^\top\right)_{ij}(t,x)\,\partial^2_{x_ix_j}.
$$

This is the *infinitesimal generator* of the diffusion $X_t$. Itô's lemma applied to $f(t,X_t)$ yields

$$
df(t,X_t) = \left(\partial_t f + \mathcal{A}f\right)dt + (\nabla_xf)^\top\sigma\,dW_t.
$$

Integrating from $t$ to $T$ and taking conditional expectations (the Itô integral term vanishes because it is a martingale with zero mean):

$$
\mathbb{E}\!\left[f(T,X_T)\,\big|\,X_t = x\right] - f(t,x) = \mathbb{E}\!\left[\int_t^T (\partial_s f + \mathcal{A}f)(s,X_s)\,ds\;\bigg|\;X_t = x\right].
$$

If $f$ solves $\partial_t f + \mathcal{A}f = 0$ (the Kolmogorov backward equation) with terminal data $f(T,\cdot) = g$, the right-hand side vanishes and we get

$$
f(t,x) = \mathbb{E}\!\left[g(X_T)\,\big|\,X_t = x\right].
$$

This is the simplest instance of the Feynman-Kac formula — without potential or source terms. The full formula, with a potential $V$ and source $f$, is established in Chapter 3 by the same argument with an additional exponential weighting.


---

## Chapter 3. The Feynman-Kac Formula

### 3.1 Setup and notation

We work on the time interval $[0,T]$ in $\mathbb{R}^d$. Fix coefficients

$$
\mu\colon [0,T]\times\mathbb{R}^d\to\mathbb{R}^d, \qquad \sigma\colon [0,T]\times\mathbb{R}^d\to\mathbb{R}^{d\times m},
$$

satisfying Lipschitz and linear growth conditions (Theorem 2.4), and consider the associated diffusion

$$
dX_s = \mu(s,X_s)\,ds + \sigma(s,X_s)\,dW_s, \qquad X_t = x.
\tag{SDE}
$$

We further fix:

- a *potential* (or *killing rate*) $V\colon [0,T]\times\mathbb{R}^d\to\mathbb{R}$, continuous, bounded below,
- a *source* $f\colon [0,T]\times\mathbb{R}^d\to\mathbb{R}$, continuous,
- *terminal data* $g\colon \mathbb{R}^d\to\mathbb{R}$, continuous and of at most polynomial growth.

Define the infinitesimal generator $\mathcal{A}$ as in Section 2.5, and consider the PDE

$$
\boxed{\partial_t u + \mathcal{A}u - V\,u + f = 0, \qquad u(T,x)=g(x).}
\tag{FK-PDE}
$$

Written out in coordinates:

$$
\partial_t u + \sum_i \mu_i\,\partial_{x_i}u + \frac{1}{2}\sum_{i,j}(\sigma\sigma^\top)_{ij}\,\partial^2_{x_ix_j}u - V\,u + f = 0.
$$

This is a *terminal-value problem* — the data is prescribed at the final time $T$ and the equation is solved backward in time. The sign conventions are such that for $\mu = 0$, $\sigma = \sqrt{2}\,I_d$, $V = 0$, $f = 0$, the equation reduces to $\partial_t u + \Delta u = 0$, i.e.\ the backward heat equation.

> **Remark (initial-value vs. terminal-value).** If one prefers an initial-value formulation, set $\tau = T-t$. Then $\tilde{u}(\tau,x) = u(T-\tau,x)$ satisfies a forward PDE $\partial_\tau\tilde{u} = \mathcal{A}\tilde{u} - V\tilde{u} + f$ with $\tilde{u}(0,x) = g(x)$. Both formulations are equivalent; the terminal-value version is more natural in the stochastic setting.


### 3.2 Statement of the theorem

**Theorem 3.1 (Feynman-Kac formula).** Under the hypotheses above, suppose $u \in C^{1,2}([0,T)\times\mathbb{R}^d)\,\cap\,C([0,T]\times\mathbb{R}^d)$ is a solution of (FK-PDE) satisfying the growth condition

$$
|u(t,x)| \le C(1+|x|^p) \quad\text{for some } C,p > 0, \text{ uniformly in } t\in[0,T].
$$

Then for every $(t,x)\in[0,T]\times\mathbb{R}^d$,

$$
\boxed{
u(t,x) = \mathbb{E}\!\left[\,g(X_T)\,e^{-\int_t^T V(r,X_r)\,dr} + \int_t^T f(s,X_s)\,e^{-\int_t^s V(r,X_r)\,dr}\,ds\;\bigg|\;X_t = x\right],
}
\tag{FK}
$$

where $X_s$ solves (SDE) with $X_t = x$.

**Notation.** The two terms have distinct roles:

- The first term $\mathbb{E}[g(X_T)\,e^{-\int_t^T V\,dr}]$ encodes the terminal condition, weighted by the *survival factor* $e^{-\int V}$.
- The second term $\mathbb{E}[\int_t^T f(s,X_s)\,e^{-\int_t^s V\,dr}\,ds]$ accumulates the source $f$ along paths, each contribution discounted by the survival factor up to the time of emission.

**Important special cases.**

| Special case | PDE | Feynman-Kac representation |
|---|---|---|
| $V=0,\;f=0$ | $\partial_t u + \mathcal{A}u = 0$ | $u(t,x) = \mathbb{E}[g(X_T)\mid X_t=x]$ |
| $f=0$ | $\partial_t u + \mathcal{A}u - Vu = 0$ | $u(t,x) = \mathbb{E}\!\left[g(X_T)\,e^{-\int_t^T V\,dr}\mid X_t=x\right]$ |
| $V=0$ | $\partial_t u + \mathcal{A}u + f = 0$ | $u(t,x) = \mathbb{E}\!\left[g(X_T) + \int_t^T f(s,X_s)\,ds\mid X_t=x\right]$ |


### 3.3 Proof of Theorem 3.1

We prove the homogeneous case ($f = 0$) in detail. The extension to $f \neq 0$ follows by the same method.

**Step 1: define the discounted process.** Set

$$
D_s := \exp\!\left(-\int_t^s V(r,X_r)\,dr\right), \qquad M_s := u(s,X_s)\,D_s, \qquad s\in[t,T].
$$

Our goal is to show that $M$ is a martingale, from which the formula follows immediately.

**Step 2: apply Itô's formula to $M_s$.** Since $D_s$ is a process of *bounded variation* (it satisfies the ODE $dD_s = -V(s,X_s)\,D_s\,ds$, with no $dW_s$ component), the product rule gives

$$
dM_s = D_s\,du(s,X_s) + u(s,X_s)\,dD_s + \underbrace{d[u(\cdot,X_\cdot),\, D]_s}_{=\;0\;\text{(no common $dW$ terms)}}.
$$

By Itô's lemma (Theorem 2.2),

$$
du(s,X_s) = \left(\partial_s u + \mathcal{A}u\right)ds + (\nabla_x u)^\top\sigma\,dW_s,
$$

and

$$
dD_s = -V(s,X_s)\,D_s\,ds.
$$

Substituting:

$$
dM_s = D_s\!\left[\partial_s u + \mathcal{A}u\right]ds + D_s\,(\nabla_x u)^\top\sigma\,dW_s + u\,(-V D_s)\,ds
$$
$$
= D_s\!\left[\partial_s u + \mathcal{A}u - V u\right]ds + D_s\,(\nabla_x u)^\top\sigma\,dW_s.
$$

**Step 3: use the PDE.** Since $u$ solves $\partial_t u + \mathcal{A}u - Vu = 0$, the $ds$-term vanishes identically:

$$
dM_s = D_s\,(\nabla_x u(s,X_s))^\top\sigma(s,X_s)\,dW_s.
$$

This is a stochastic integral with respect to $dW_s$, hence a local martingale.

**Step 4: verify the martingale property.** The polynomial growth condition on $u$, combined with the Lipschitz/linear growth conditions on $\mu$ and $\sigma$ (which imply moment bounds $\mathbb{E}[\sup_{t\le s\le T}|X_s|^p]<\infty$) and the boundedness below of $V$ (which ensures $D_s$ is bounded), guarantee that

$$
\mathbb{E}\!\left[\int_t^T \left|D_s\,(\nabla_xu)^\top\sigma\right|^2 ds\right] < \infty.
$$

Hence $M$ is a *true* martingale (not merely a local martingale). For the details, see Øksendal [2, §8.2] or Karatzas and Shreve [3, §4.4].

**Step 5: conclude.** The martingale property gives $\mathbb{E}[M_T \mid \mathcal{F}_t] = M_t$, i.e.,

$$
\mathbb{E}\!\left[\,u(T,X_T)\,\exp\!\left(-\!\int_t^T V(r,X_r)\,dr\right)\;\bigg|\; X_t = x\right] = u(t,x)\cdot 1.
$$

Using the terminal condition $u(T,X_T) = g(X_T)$:

$$
u(t,x) = \mathbb{E}\!\left[\,g(X_T)\,\exp\!\left(-\!\int_t^T V(r,X_r)\,dr\right)\;\bigg|\; X_t = x\right]. \qquad\square
$$

**Extension to $f\neq 0$.** When $\partial_tu + \mathcal{A}u -Vu + f = 0$, the PDE no longer kills the $ds$-term entirely; instead one gets $dM_s = D_s\,f(s,X_s)\,ds + (\text{martingale term})$. Integrating from $t$ to $T$, taking expectations, and rearranging yields the full formula (FK).

### 3.4 Probabilistic interpretation of the potential term

When $V \ge 0$, the factor $\exp(-\int_t^T V(r,X_r)\,dr) \in [0,1]$ has a natural interpretation: it is the *survival probability* of a particle that diffuses according to the SDE and is *killed* (absorbed) at rate $V(x)$ when located at position $x$. Paths that traverse regions of high potential $V$ are exponentially down-weighted. In the language of mathematical physics, the operator $-\mathcal{A} + V$ is a *Schrödinger operator*, and its semigroup is generated by *killed Brownian motion*.

### 3.5 The dictionary: PDE coefficients ↔ SDE coefficients

The Feynman-Kac formula establishes a systematic correspondence. Given a PDE of the form

$$
\partial_t u + \sum_i b_i\,\partial_{x_i}u + \frac{1}{2}\sum_{i,j}a_{ij}\,\partial^2_{x_ix_j}u - V u + f = 0,
$$

the associated SDE is

$$
dX_s = b(s,X_s)\,ds + \sigma(s,X_s)\,dW_s \qquad\text{where}\quad \sigma\sigma^\top = a.
$$

| PDE term | Role | SDE counterpart |
|---|---|---|
| $b_i\,\partial_{x_i}u$ | advection / first-order transport | drift $\mu = b$ |
| $\frac{1}{2}a_{ij}\,\partial^2_{x_ix_j}u$ | diffusion / second-order | diffusion matrix, $\sigma\sigma^\top = a$ |
| $-V u$ | absorption / reaction / potential | killing rate $V$ along paths |
| $f$ | source / forcing | accumulated source along paths |
| $g(x) = u(T,x)$ | terminal data | terminal evaluation $g(X_T)$ |

**Finding $\sigma$ from $a$.** The PDE specifies the *diffusion tensor* $a = \sigma\sigma^\top$. One needs a "square root": any $\sigma$ satisfying $\sigma\sigma^\top = a$ works (the law of the SDE solution depends on $a = \sigma\sigma^\top$, not on $\sigma$ individually). If $a$ is a scalar multiple of the identity, $a = 2\alpha\,I_d$, then $\sigma = \sqrt{2\alpha}\,I_d$.

### 3.6 Application: product-of-sines heat equation

Consider the PDE from our problem set:

$$
\partial_t u + \alpha\,\Delta u = 0, \qquad u(T,x) = \prod_{i=1}^d \sin(a_i\,x_i).
$$

**Reading off the SDE.** Comparing with (FK-PDE): $\mu = 0$, $\frac{1}{2}\sigma\sigma^\top = \alpha\,I_d$ so $\sigma = \sqrt{2\alpha}\,I_d$, $V = 0$, $f = 0$. The associated SDE is

$$
dX_s = \sqrt{2\alpha}\,dW_s, \qquad X_t = x,
$$

i.e., $X_s = x + \sqrt{2\alpha}\,(W_s - W_t)$. The Feynman-Kac formula gives:

$$
u(t,x) = \mathbb{E}\!\left[\prod_{i=1}^d \sin\!\left(a_i\bigl(x_i + \sqrt{2\alpha}\,(W^i_T - W^i_t)\bigr)\right)\right].
$$

Since the $d$ components of Brownian motion are independent, this factorises:

$$
u(t,x) = \prod_{i=1}^d \mathbb{E}\!\left[\sin\!\left(a_i x_i + a_i\sqrt{2\alpha}\,Z_i\sqrt{T-t}\right)\right],
$$

where $Z_i \sim \mathcal{N}(0,1)$ are i.i.d. Using the identity $\mathbb{E}[\sin(\theta + c Z)] = \sin(\theta)\,e^{-c^2/2}$ (which follows from the characteristic function of the Gaussian), each factor gives $\sin(a_ix_i)\,e^{-\alpha a_i^2(T-t)}$, so

$$
u(t,x) = \left(\prod_{i=1}^d \sin(a_i x_i)\right)e^{-\alpha|a|^2(T-t)},
$$

which matches the known analytic solution (after the substitution $T-t \to t$ if one uses the forward convention).

### 3.7 Application: Gauss diffusion with advection

The PDE is

$$
\partial_t u + a\,(\nabla u \cdot v) + b\,\Delta u = 0,
$$

with constant $a, b$ and constant vector $v \in \mathbb{R}^d$.

**Reading off the SDE.** We have $\mu = a\,v$, $\frac{1}{2}\sigma\sigma^\top = b\,I_d$ so $\sigma = \sqrt{2b}\,I_d$, $V = 0$, $f = 0$. The SDE is

$$
dX_s = a\,v\,ds + \sqrt{2b}\,dW_s, \qquad X_t = x.
$$

This is Brownian motion with constant drift: $X_s = x + a\,v\,(s-t) + \sqrt{2b}\,(W_s - W_t)$. The Feynman-Kac formula gives

$$
u(t,x) = \mathbb{E}\!\left[\,g\!\left(x + a\,v\,(T-t) + \sqrt{2b}\,W_{T-t}\right)\right],
$$

where $g$ is the initial/terminal condition. With $g(x) = (2\pi)^{-d/2}e^{-|x|^2/2}$ (the Gaussian IC), this is a convolution of two Gaussians, which can be evaluated analytically.

### 3.8 Application: travelling Gaussian with potential and source

The PDE is

$$
\partial_t u - \delta\,\Delta u + \vec{v}\cdot\nabla u + w\,u = f,
$$

where $w = -2\delta\alpha\sum a_i^2 < 0$ (note the sign: $w$ is **negative** for this problem, acting as a growth term rather than a killing term).

**Reading off the SDE.** Rewrite in the form $\partial_t u + \mathcal{A}u - Vu + f = 0$:

$$
\partial_t u + (-\vec{v})\cdot\nabla u + \delta\,\Delta u - w\,u + f = 0,
$$

so $\mu = -\vec{v}$, $\frac{1}{2}\sigma\sigma^\top = \delta\,I_d$ giving $\sigma = \sqrt{2\delta}\,I_d$, and the potential is $V = w < 0$. The SDE is

$$
dX_s = -\vec{v}\,ds + \sqrt{2\delta}\,dW_s, \qquad X_t = x.
$$

The Feynman-Kac representation is:

$$
u(t,x) = \mathbb{E}\!\left[\,g(X_T)\,e^{-w(T-t)} + \int_t^T f(s,X_s)\,e^{-w(s-t)}\,ds\;\bigg|\;X_t = x\right],
$$

where the constant potential $V = w$ simplifies the exponential. Since $w < 0$, the factor $e^{-w(T-t)} = e^{|w|(T-t)} > 1$ is a *growth* factor (not decay). The source term $f$ is negative and partially compensates this growth, yielding a well-behaved solution.

> **Remark (variance).** When $|w|$ is large (e.g. high dimension or large $\delta\alpha$), the growth factor $e^{|w|T}$ causes exponential variance blow-up in the Monte Carlo estimator, even though the true solution remains bounded. For practical computation one should keep $|w|T$ moderate, or employ importance sampling via Girsanov's theorem (Section 3.11) to tilt the SDE drift and dramatically reduce variance.

### 3.9 Application: Fokker-Planck and the molecular configuration problem

The Fokker-Planck PDE for the molecular system with interatomic potential $U$ reads:

$$
\partial_t\,p - D\,\Delta p + \frac{1}{\xi}\nabla p \cdot \nabla U + \frac{1}{\xi}p\,\Delta U = 0,
$$

where $p$ is a probability density, $\xi$ is a friction coefficient, and $D = k_BT/\xi$ is the diffusion constant.

**Rewriting.** Observe that $\nabla p \cdot \nabla U + p\,\Delta U = \nabla\cdot(p\,\nabla U)$ — this is a divergence-form term. One can equivalently write this PDE in the form:

$$
\partial_t p + \underbrace{\frac{1}{\xi}(\nabla U)\cdot\nabla p}_{\text{advection}} + \underbrace{D\,(-\Delta p)}_{\text{anti-diffusion?}} + \underbrace{\frac{1}{\xi}(\Delta U)\,p}_{\text{potential}} = 0.
$$

Wait — let us be careful with signs. The PDE as stated is:

$$
\partial_t p = D\,\Delta p - \frac{1}{\xi}\nabla p \cdot \nabla U - \frac{1}{\xi}p\,\Delta U.
$$

This is a **forward** (Fokker-Planck) equation for the probability density of the Langevin dynamics

$$
dX_t = -\frac{1}{\xi}\nabla U(X_t)\,dt + \sqrt{2D}\,dW_t.
$$

The Feynman-Kac formula in its standard form applies to the *backward* equation for conditional expectations. However, the key observation for the molecular problem is the following:

**The SDE itself is what we simulate.** To compute quantities like mean first-passage times between molecular configurations $A$ and $B$ in the 21-dimensional configuration space, one does not need to solve the Fokker-Planck PDE on a grid. Instead:

1. **Simulate** the 21-dimensional Langevin SDE starting from configuration $A$:
$$
X_{k+1} = X_k - \frac{\Delta t}{\xi}\nabla U(X_k) + \sqrt{2D\,\Delta t}\;\xi_k, \qquad \xi_k \sim \mathcal{N}(0,I_{21}).
$$
2. **Record** the first time $\tau$ that $X_t$ enters the target set $B$.
3. **Repeat** for $N$ independent trajectories and estimate $\mathbb{E}[\tau] \approx \frac{1}{N}\sum_{i=1}^N \tau^{(i)}$.

This is a direct application of the Feynman-Kac philosophy: replace PDE computation with Monte Carlo sampling of SDE paths. The cost scales *linearly* in $d = 21$ (each Euler-Maruyama step costs $\mathcal{O}(d)$) rather than exponentially.

> **Remark.** For *rare events* — when the passage from $A$ to $B$ involves crossing a high energy barrier and $\mathbb{E}[\tau]$ is astronomically large — naive Monte Carlo becomes impractical. Advanced techniques such as *importance sampling* (via Girsanov's theorem to tilt the SDE drift toward the target), *transition path sampling*, or *metadynamics* are needed. These are active areas of research in computational chemistry and belong to the broader framework of *rare-event simulation* [7, 8].

### 3.10 Monte Carlo estimation and convergence

**Algorithm (basic Feynman-Kac Monte Carlo).** To compute $u(t,x)$ for the PDE (FK-PDE):

1. Choose a time step $\Delta t$ and set $n = \lceil(T-t)/\Delta t\rceil$.
2. For $i = 1, \dots, N$ (independent samples):
   - Set $X_0^{(i)} = x$.
   - For $k = 0, \dots, n-1$: advance by the Euler-Maruyama scheme
     $$X_{k+1}^{(i)} = X_k^{(i)} + \mu(t_k, X_k^{(i)})\,\Delta t + \sigma(t_k, X_k^{(i)})\,\sqrt{\Delta t}\;\xi_k^{(i)}, \quad \xi_k^{(i)}\sim\mathcal{N}(0,I_d).$$
   - Compute the path functional:
     $$\Phi^{(i)} = g(X_n^{(i)})\,\exp\!\left(-\Delta t\sum_{k=0}^{n-1}V(t_k,X_k^{(i)})\right) + \Delta t\sum_{k=0}^{n-1}f(t_k,X_k^{(i)})\exp\!\left(-\Delta t\sum_{j=0}^{k-1}V(t_j,X_j^{(i)})\right).$$
3. Estimate $u(t,x) \approx \hat{u}_N = \frac{1}{N}\sum_{i=1}^N \Phi^{(i)}$.

**Error analysis.** The total error decomposes as

$$
|u(t,x) - \hat{u}_N| \;\le\; \underbrace{|u(t,x) - u_{\Delta t}(t,x)|}_{\text{time-discretisation bias}} + \underbrace{|u_{\Delta t}(t,x) - \hat{u}_N|}_{\text{Monte Carlo statistical error}}.
$$

- **Bias.** The Euler-Maruyama scheme has *weak convergence* of order 1: the bias is $\mathcal{O}(\Delta t)$. Higher-order schemes exist (e.g. Milstein for $d=1$).
- **Statistical error.** By the central limit theorem, $\hat{u}_N - u_{\Delta t} \approx \mathcal{N}(0, \sigma^2_\Phi / N)$, giving the dimension-independent rate $\mathcal{O}(N^{-1/2})$.

**Variance reduction** is essential for practical efficiency. Key techniques include:

- *Antithetic variates:* for each path driven by increments $\xi_k$, also simulate the path with $-\xi_k$.
- *Control variates:* subtract a correlated quantity with known mean.
- *Multilevel Monte Carlo (MLMC)* [6]: compute expectations as a telescoping sum across time-discretisation levels, achieving the same accuracy at lower total cost.
- *Importance sampling:* change the probability measure via Girsanov's theorem to tilt the SDE drift toward regions that contribute most to the expectation, reweighting paths by the likelihood ratio. This is especially effective when the survival factor $e^{-\int V}$ grows exponentially (see Section 3.11).


### 3.11 Importance sampling via Girsanov's theorem

When the potential $V$ is negative (a growth term), the survival factor $e^{-\int_0^T V\,ds}$ grows exponentially, causing extreme variance in the MC estimator even though the true solution $u$ remains bounded. Importance sampling (IS) addresses this by changing the probability measure under which paths are sampled, reweighting each path by a likelihood ratio (the *Girsanov weight*) so that the expectation is preserved but the variance is reduced.

#### Girsanov's theorem

**Theorem 3.2 (Girsanov).** Let $\theta\colon [0,T]\times\mathbb{R}^d \to \mathbb{R}^d$ be an adapted process satisfying Novikov's condition $\mathbb{E}[\exp(\frac{1}{2}\int_0^T|\theta_s|^2\,ds)] < \infty$. Define the *Girsanov weight*

$$
G_T := \exp\!\left(-\int_0^T \theta_s \cdot dW_s - \frac{1}{2}\int_0^T |\theta_s|^2\,ds\right).
$$

Then under the new probability measure $\mathbb{Q}$ defined by $d\mathbb{Q}/d\mathbb{P} = G_T$, the process

$$
\widetilde{W}_s := W_s + \int_0^s \theta_r\,dr
$$

is a standard Brownian motion.

Equivalently: the original SDE $dX_s = \mu\,ds + \sigma\,dW_s$ becomes, under $\mathbb{Q}$,

$$
dX_s = (\mu + \sigma\theta)\,ds + \sigma\,d\widetilde{W}_s.
$$

The drift has been *tilted* by $\sigma\theta$, and $\widetilde{W}$ is the $\mathbb{Q}$-Brownian motion.

#### IS formula for the Feynman-Kac representation

Applying the change of measure to the FK formula (with $G_s$ denoting the Girsanov weight accumulated up to time $s$):

$$
\boxed{
u(t,x) = \mathbb{E}_{\mathbb{Q}}\!\left[\,g(X_T)\,e^{-\int_0^T V\,ds}\,G_T + \int_0^T f(s,X_s)\,e^{-\int_0^s V\,dr}\,G_s\,ds\;\bigg|\;X_0 = x\right],
}
$$

where $X_s$ now follows the tilted SDE with drift $\mu + \sigma\theta$ and the Girsanov log-weight is accumulated as

$$
\log G_s = -\int_0^s \theta_r \cdot d\widetilde{W}_r - \frac{1}{2}\int_0^s |\theta_r|^2\,dr.
$$

The expectation is unchanged (by Girsanov's theorem), but the *variance* depends on $\theta$.

#### Optimal drift

**Proposition.** The choice $\theta^*(s,x) = \sigma\,\nabla\log u(T-s,x)$ gives **zero variance** in the terminal-value estimator (when $f=0$).

*Proof sketch.* With this choice, the Girsanov weight exactly cancels the variance of $g(X_T)\,e^{-\int V}$: the integrand becomes deterministic. See [9, §6.2] for a full proof.

In practice $u$ is unknown (it is what we are trying to compute), so the optimal drift cannot be used directly. Practical strategies include:

- **Pilot runs:** use a coarse MC estimate to approximate $\nabla\log u$, then run a refined IS estimator.
- **Parametric fits:** approximate $\log u$ with a parametric family (e.g. quadratic) and optimise the parameters.
- **Known solutions:** when the exact solution is available (as for the travelling Gaussian), use the exact $\nabla\log u$ as a benchmark to validate the IS implementation and measure the achievable variance reduction.

#### Application to the travelling Gaussian

For the travelling Gaussian (Section 3.8) with known solution $u(t,x) = \exp(-\alpha\sum(a_ix_i-b_i+c_it)^2)\,e^{-\beta t}\cos(\gamma t)$, we have

$$
\nabla_x \log u = -2\alpha\,a_i(a_ix_i - b_i + c_i\tau), \qquad \tau = t_{\mathrm{PDE}},
$$

so the optimal IS drift is $\theta^* = \sigma\,\nabla\log u = \sqrt{2\delta}\,(-2\alpha)\,a_i(a_ix_i - b_i + c_i\tau)$.

With this drift, the MC estimator achieves near-zero variance regardless of the magnitude of $|w|T$, eliminating the exponential variance blow-up described in Section 3.8.

> **Remark.** The optimal IS drift depends on the PDE time $\tau = T - s$ (where $s$ is SDE time), not on SDE time directly. When implementing, one must be careful to evaluate $\nabla\log u$ at the correct PDE time corresponding to each SDE step.


---

## References

[1] M. Kac, "On distributions of certain Wiener functionals," *Trans. Amer. Math. Soc.*, vol. 65, no. 1, pp. 1–13, 1949.

[2] B. Øksendal, *Stochastic Differential Equations: An Introduction with Applications*, 6th ed. Springer, 2003. (The Feynman-Kac formula is in §8.2.)

[3] I. Karatzas and S. E. Shreve, *Brownian Motion and Stochastic Calculus*, 2nd ed. Springer, 1991. (Feynman-Kac in §4.4.)

[4] L. C. Evans, *Partial Differential Equations*, 2nd ed. AMS, 2010. (§4.3 connects the heat equation to Brownian motion.)

[5] E. Pardoux and S. Peng, "Adapted solution of a backward stochastic differential equation," *Systems & Control Letters*, vol. 14, no. 1, pp. 55–61, 1990. (Nonlinear Feynman-Kac via BSDEs.)

[6] M. B. Giles, "Multilevel Monte Carlo path simulation," *Operations Research*, vol. 56, no. 3, pp. 607–617, 2008.

[7] J. Han, A. Jentzen, and W. E, "Solving high-dimensional partial differential equations using deep learning," *Proc. Natl. Acad. Sci.*, vol. 115, no. 34, pp. 8505–8510, 2018. (Deep BSDE method for high-dimensional nonlinear PDEs.)

[8] D. W. Stroock and S. R. S. Varadhan, *Multidimensional Diffusion Processes*. Springer, 1979.

[9] P. Glasserman, *Monte Carlo Methods in Financial Engineering*. Springer, 2003. (Importance sampling and Girsanov's theorem in §4.6–4.7.)
