# My summary

- Every method below is essentially a different strategy to *avoid* building the grid.

## MC / Feynman Kac
- he solution u(x,t) of a parabolic PDE is an *expectation* of a functional over stochastic trajectories
- build paths in time and average
- takes in x,t -> computes to get a solution

## sparse grid
- if smooth function
- exploits the fact the minimal high dim mixing

## tensor trains
- chain of low rank tensors
- represent all in TT format
- good if correlations between dims decay fast - heat eq

## PINNS
- for d > 10 needs further tricks
- slow convergence for stiff systems

## deep Galerkin / Ritz method
- pinn with weak form - variational / minimze en
- var form better conditioned than the pde

## deep BSDE
- NN as the grad u times smthg
- almost like MC
- simulates forward paths
- read original 2018 paper - shows in 100 dims



---



# Numerical Methods for High-Dimensional Diffusion-Like PDEs

The central enemy here is the **curse of dimensionality**: a grid with $N$ points per axis in $d$ dimensions needs $N^d$ points. At $d=6$, even $N=100$ gives $10^{12}$ points — hopeless. Every method below is essentially a different strategy to *avoid* building that grid.

---

## 1. Monte Carlo / Feynman–Kac Methods

### Conceptual summary
The solution $u(x,t)$ of a parabolic PDE is an *expectation* of a functional over stochastic trajectories. You never build a grid; you sample paths.

### For the student

The **Feynman–Kac formula** connects PDEs to SDEs. For the heat equation
$$\partial_t u = \tfrac{1}{2}\Delta u, \quad u(x,T) = g(x)$$
the solution at $(x,t)$ is
$$u(x,t) = \mathbb{E}[g(X_T) \mid X_t = x]$$
where $X_s$ solves $dX_s = dW_s$ (Brownian motion). More generally, for
$$\partial_t u + \mathcal{L}u + f = 0$$
with $\mathcal{L}$ the generator of an Itô diffusion $dX = \mu\,dt + \sigma\,dW$, the solution is an expectation over paths of $X$.

**Algorithm:** (1) Start $M$ particles at $x$. (2) Simulate SDE paths forward to $T$ using Euler–Maruyama: $X_{k+1} = X_k + \mu \Delta t + \sigma \sqrt{\Delta t}\, \xi_k$, $\xi_k \sim \mathcal{N}(0,I)$. (3) Average $g(X_T)$ over all particles.

**Scaling:** Cost is $O(M \cdot T/\Delta t \cdot d)$ — *linear in $d$*. This is the killer advantage. Error goes as $1/\sqrt{M}$ regardless of $d$.

**Catch:** You get $u$ at one point $x$ at a time. For the Fokker–Planck equation (governing the *density* rather than a single trajectory), you run the forward process and bin particles — but binning reintroduces the curse unless you're clever.

**Relevant for:** heat eq, Kolmogorov backward eq, Black–Scholes, quantum imaginary-time evolution.

---

## 2. Sparse Grids (Smolyak Construction)

### Conceptual summary
Instead of the full tensor-product grid, use a clever *sparse* combination of low-resolution grids along each axis that retains most accuracy while reducing cost from $O(N^d)$ to $O(N \log^{d-1} N)$.

### For the student

The idea: build 1D hierarchical basis functions $\phi_{l,i}(x)$ (hat functions or wavelets at level $l$, node $i$). The full grid uses all combinations. The Smolyak rule keeps only combinations where $\sum_j l_j \leq d + k$ for some small $k$, exploiting the fact that mixed high-order interactions are small for smooth functions.

If $u$ has bounded **mixed partial derivatives** (the relevant smoothness class), the error of the sparse grid approximation is
$$\|u - u_{\text{sparse}}\| = O(N^{-r} \log^{(d-1)r} N)$$
vs. $O(N^{-r})$ for the full grid — but with $O(N \log^{d-1} N)$ points instead of $O(N^d)$.

**Practical limit:** Works well up to $d \approx 10$–$20$ if $u$ is smooth with small mixed derivatives. Fails if $u$ has strong cross-coupling (e.g., strongly correlated Fokker–Planck).

**Software:** `Tasmanian`, `SG++`, `SPARSE_GRID` (MATLAB).

**Connection to ML:** Sparse grids are related to ANOVA decompositions and to why random forests generalize: both implicitly assume weak interaction between features.

---

## 3. Tensor Train / Tensor Network Methods

### Conceptual summary
Represent the $d$-dimensional solution as a *chain of low-rank matrices* (the tensor train / MPS format). If the solution has low tensor rank, storage and computation scale *linearly* in $d$.

### For the student

A $d$-dimensional array $u[i_1, \ldots, i_d]$ (each index $1,\ldots,N$) normally needs $N^d$ numbers. In **tensor train (TT) format**:
$$u[i_1,\ldots,i_d] = G_1[i_1]\, G_2[i_2] \cdots G_d[i_d]$$
where each $G_k[i_k]$ is an $r_{k-1} \times r_k$ matrix (with $r_0 = r_d = 1$). Storage: $O(d N r^2)$ where $r$ is the max bond dimension / rank. For $d=7, N=100, r=20$: about $2.8\times 10^5$ numbers instead of $10^{14}$.

**For PDEs:** Discretize the PDE operator $L$ as a matrix and represent it in TT format too (TT-operator). Then solve the linear system $Lu = f$ with DMRG-style alternating least squares: fix all cores except one, solve the small local problem, sweep through all cores, repeat until convergence.

**Key insight from physics:** This is exactly the DMRG algorithm used in quantum chemistry and condensed matter. The PDE is formally equivalent to a quantum many-body problem on a 1D chain.

**When it works:** When the solution has *low entanglement* — i.e., its correlations between groups of dimensions decay fast. For isotropic heat equations from smooth initial conditions, ranks stay low. For highly anisotropic or multiscale problems, ranks blow up.

**Software:** `TT-Toolbox` (MATLAB/Python), `ttpy`, `ITensor`, `tntorch`.

---

## 4. Physics-Informed Neural Networks (PINNs)

### Conceptual summary
Parameterize $u(x,t)$ by a neural network, then minimize a loss that penalizes the PDE residual, boundary conditions, and initial conditions — all evaluated at randomly sampled collocation points.

### For the student

$$\mathcal{L} = \underbrace{\frac{1}{N_r}\sum_i \left|\partial_t u_\theta(x_i,t_i) - \mathcal{L}u_\theta(x_i,t_i)\right|^2}_{\text{PDE residual}} + \underbrace{\lambda_\text{BC} \cdot \text{BC loss} + \lambda_\text{IC} \cdot \text{IC loss}}_{\text{boundary/initial conditions}}$$

Derivatives $\partial_t u_\theta$, $\Delta u_\theta$ etc. are computed via **automatic differentiation** — exactly what you do in any ML training loop, just backprop through the network with respect to *inputs*, not weights.

**Training = optimization** over $\theta$ via Adam/L-BFGS. No grid needed; points $x_i$ are sampled randomly from the domain (Monte Carlo quadrature), so cost scales benignly with $d$.

**Strengths:** Meshfree; easy to code (just PyTorch/JAX); handles complex geometries; can incorporate noisy data (inverse problems).

**Weaknesses:** Slow convergence for multiscale / stiff problems; sensitive to hyperparameters and loss weighting; no rigorous error control; for high-$d$ the curse reappears in the required expressiveness of the network and the number of collocation points. **Not yet reliable for $d > 10$** without further tricks.

**Key papers:** Raissi et al. (2019) JCP; Müller & Zeinhofer (2023) for convergence theory.

**Schrodinger note:** PINNs can handle the time-dependent Schrödinger eq by treating real and imaginary parts separately, but quantum chemistry at high $d$ is much better served by VMC (see below).

---

## 5. Deep Galerkin Method (DGM) & Deep Ritz

### Conceptual summary
Reformulate the PDE as a variational (energy minimization) or least-squares problem, then use a neural network as a trial function space and optimize directly.

### For the student

**Deep Ritz** (E & Yu 2018): For a PDE derived from minimizing an energy functional $\mathcal{E}[u]$ (e.g., $-\Delta u = f$ minimizes $\int |\nabla u|^2 - 2fu$), directly minimize $\mathcal{E}[u_\theta]$ where $u_\theta$ is a neural network. Gradient estimated by Monte Carlo. Cost: $O(d)$ per gradient evaluation.

**DGM** (Sirignano & Spiliopoulos 2018): Replace the PDE residual minimization in PINNs with a Galerkin-like projection. Uses a specialized recurrent-style architecture and time-marching.

**Why better than vanilla PINNs in theory:** Variational formulations are often better conditioned than strong-form residuals; the energy landscape is smoother.

**Connection to ML you know:** This is literally ERM (empirical risk minimization) where the "data" are quadrature points and the "label" is zero residual. The generalization gap IS the quadrature error.

---

## 6. Deep BSDE Method

### Conceptual summary
Reformulate the PDE as a *backward stochastic differential equation* (BSDE) and learn the solution at $t=0$ by training a neural network to match terminal conditions along forward SDE trajectories.

### For the student

Every semilinear parabolic PDE
$$\partial_t u + \tfrac{1}{2}\text{tr}(\sigma\sigma^\top D^2 u) + \mu \cdot \nabla u + f(x,u,\nabla u) = 0, \quad u(x,T) = g(x)$$
has a BSDE representation: find processes $(Y_t, Z_t)$ (think: $Y_t = u(X_t,t)$, $Z_t = \sigma^\top \nabla u(X_t,t)$) satisfying
$$dY_t = -f(X_t, Y_t, Z_t)\,dt + Z_t^\top dW_t, \quad Y_T = g(X_T).$$

**Algorithm** (E, Han, Jentzen 2017): Parameterize $Z_t$ by neural networks $Z_\theta(X_t, t)$, and learn the initial value $Y_0$ as a trainable scalar. Simulate $M$ forward paths $X$, march the BSDE forward, and minimize the loss $\mathbb{E}[|Y_T - g(X_T)|^2]$ over $\theta$ and $Y_0$.

**Scaling:** Cost $O(M \cdot T/\Delta t)$ with each step requiring a forward pass through $Z_\theta$ — linear in $d$. This solved 100-dimensional Hamilton–Jacobi–Bellman equations in the original 2018 paper — a landmark result.

**Catch:** Works best for *semilinear* equations. Fully nonlinear PDEs require second-order BSDEs (2BSDEs), which are harder. Convergence theory is incomplete.

**Highly recommended** as a first serious method to implement for Fokker–Planck adjoint problems or optimal control.

---

## 7. Variational Monte Carlo (VMC) — especially for Schrödinger

### Conceptual summary
Parameterize the wavefunction / probability density by a neural network (or other ansatz), and optimize energy/free energy by Monte Carlo sampling from the current distribution.

### For the student

For the *time-independent* Schrödinger eq $H\psi = E\psi$, minimize the variational energy:
$$E[\psi_\theta] = \frac{\langle \psi_\theta | H | \psi_\theta \rangle}{\langle \psi_\theta | \psi_\theta \rangle} = \mathbb{E}_{x \sim |\psi_\theta|^2}\left[\frac{H\psi_\theta(x)}{\psi_\theta(x)}\right]$$

The expectation is estimated by MCMC (Metropolis) sampling from $|\psi_\theta|^2$. Modern ansätze: **FermiNet**, **PauliNet**, **NQS** (neural quantum states with RBMs). For classical stat mech / Fokker–Planck stationary distributions, same idea with free energy in place of $E$.

**The magic:** You never represent $\psi$ on a grid. You only ever *evaluate* it at sampled points. Cost: $O(d)$ per MCMC step (assuming sparse Hamiltonian).

**For Fokker–Planck steady state** $\nabla \cdot (D\nabla \rho + \rho \nabla V) = 0$: minimize the free energy $F[\rho] = \int \rho \log \rho + \rho V$ using a normalizing flow or energy-based model for $\rho$.

**Connection to score-based diffusion models:** The score $\nabla \log \rho$ appears in both Fokker–Planck and in denoising diffusion. Score matching *is* a way to learn Fokker–Planck solutions.

---

## 8. Diffusion Maps & Data-Driven Methods

### Conceptual summary
If you have data sampled from the invariant measure of a Fokker–Planck / Langevin system, use kernel methods to approximate the generator and solve the eigenvalue problem intrinsically on the data manifold.

### For the student

Build a kernel matrix $K_{ij} = k(x_i, x_j)$ (e.g., Gaussian) on $N$ data points. The **diffusion map** algorithm of Coifman & Lafon (2006) normalizes this to approximate the Laplace–Beltrami operator on the data manifold. Eigenvectors give you low-dimensional coordinates; eigenvalues give relaxation timescales.

For Fokker–Planck: the generator $\mathcal{L}f = -\nabla V \cdot \nabla f + \Delta f$ can be approximated as a sparse matrix on the data graph. This is the basis of **EDMD** (extended dynamic mode decomposition) and **kernel EDMD**, which approximate the Koopman/Perron–Frobenius operators.

**Scaling:** $O(N^2)$ to $O(N^3)$ in data points, independent of $d$ — because the data *already* lives on a low-dimensional manifold (if it does). Fails if the intrinsic dimension is high.

---

## Quick Comparison Table

| Method | Scales to high $d$? | Type of equation | Main limitation |
|---|---|---|---|
| Feynman–Kac MC | ✅ very well | linear parabolic | pointwise evaluation only |
| Sparse grids | ⚠️ up to ~$d=20$ | general | needs smooth, weakly coupled $u$ |
| Tensor Train | ✅ if low rank | linear, discretizable | rank blowup for entangled problems |
| PINNs / Deep Ritz | ⚠️ moderate | semilinear | slow, fragile, no error control |
| Deep BSDE | ✅ well | semilinear parabolic | theory incomplete, BC handling |
| VMC / NQS | ✅ very well | Schrödinger, stationary FP | MCMC mixing, gradients through sampling |
| Diffusion maps | ✅ if low intrinsic dim | linear FP / Koopman | needs data, quadratic in $N$ |

---

## Recommended Learning Path

If you're starting out, I'd suggest this order: **(1)** implement Feynman–Kac for the heat equation — you'll immediately feel why MC is powerful in high $d$. **(2)** Read the Deep BSDE paper (E, Han, Jentzen 2018) and implement it for a known HJB equation. **(3)** Read the tensor train DMRG-for-PDEs literature (Oseledets, Dolgov). **(4)** Depending on your application, branch into VMC (physics/quantum) or score-based methods (sampling/FP).

The field is genuinely open — there's no method that dominates above $d \approx 10$ for general equations, which is why it's an exciting research area.