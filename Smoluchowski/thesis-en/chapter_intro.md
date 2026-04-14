# The Smoluchowski equation


Consider the motion of an overdamped Brownian particle in an external potential $V(x)$ on $\mathbb{R}^d$, whose stochastic differential equation has the form

$$
dX_t = -\nabla V(X_t)\,dt + \sqrt{2\beta^{-1}}\,dW_t
$$ 

where $\beta = 1/(k_b T)$, and $k_B$ is the Bolztman constant ($\approx 10^{-23}\,J/K$).

The corresponding probability density $\rho(x,t)$ then satisfies the so-called Smoluchowski equation【14†L8575-L8581】【14†L8586-L8593】: 

$$
\partial_t \rho = \beta^{-1}\Delta\rho + \nabla\cdot\bigl(\rho\nabla V\bigr) \\
\rho(x,0)=\rho_0(x)
$$ 

This is a linear drift-diffusion equation.

In particular, the diffusion is time-reversible and admits a Gibbs invariant measure.


If $V$ is confining, that is:
【14†L8598-L8602】

$$
\lim_{|x|\to\infty}V(x)=+\infty\quad \text{and} \quad e^{-\beta V}\in L^1
$$


then there exists a unique stationary solution in the form of a Boltzman density
$$
\rho_\infty(x) = \frac{1}{Z} e^{-\beta V(x)} \qquad Z=\int_{\mathbb R^d}e^{-\beta V(x)}dx
$$ 

and under mild smoothness and convexity assumptions on $V$, this convergence is exponentially fast
【31†L9081-L9084】.



The equation can be rewritten into the form
$$
\partial_t\rho = -\nabla\cdot J,\qquad J(\rho) = -(\beta^{-1}\nabla\rho + \rho\nabla V)
$$ 
showing it is a mass-conserving convection-diffusion equation. 

The conservative (finite-volume friendly) form is obtained by introducing the probability flux $J(\rho)$.
This matches the “flux operator” viewpoint emphasized in the Smoluchowski diffusion lecture notes, where boundary conditions are naturally expressed via the normal component of flux.


## Tranforming the equation

(preconditioning via $h=\rho/\rho_\infty$)

factor out the equilibrium density:
$$
\rho(x,t)=h(x,t)\,\rho_\infty(x).
\tag{E8}
$$

(Pavliotis) $h$ then solves a backward Kolmogorov/advection–diffusion equation:
$$
\partial_t h = -\nabla V\cdot \nabla h+\beta^{-1}\Delta h,
\qquad h(\cdot,0)=\rho_0/\rho_\infty.
\tag{E9}
$$

(E9)
- removes the stiff “$\rho\nabla V$” multiplicative coupling
- in many discretizations it improves conditioning because the equilibrium is now the constant state $h\equiv 1$


The **self-adjoint (Witten Laplacian) form** applies the ground-state transformation $\rho = e^{-V/(2\varepsilon)}\psi$, converting the Fokker-Planck operator to the Schrödinger-type operator $\Delta_{V,\varepsilon}^{(0)} = -\varepsilon\Delta + \frac{|\nabla V|^2}{4\varepsilon} - \frac{\Delta V}{2}$, which is self-adjoint in $L^2(\mathbb{R}^d)$. This transformation is central to both spectral methods and the rigorous analysis of metastability.


## Boundary and Initial Conditions

### Domain

The PDE lives on $\Omega=\mathbb{R}^d$. But in the context of numerical solvers, we use a large box $\Omega=[-L,L]^d$. We set domain such that $\rho_\infty$ is neglegible on $\partial\Omega$.

### Boundary conditions

Two standard choices are  

- Dirichlet
- Neumann

Either way, one ensures $\rho$ decays at the edges.  

#### Dirichlet (reactive / absorbing)

$$
\rho(x,t)=0\quad\text{on}\:\:\partial\Omega
\tag{BC-D}
$$

The boundary acts as a perfect sink (e.g., an absorbing target or chemical reaction site), meaning particles are removed from the system instantly upon contact, thus slowly decreasing the total mass - mass leaks out.
Care is needed if one wants to track probability accurately; one might normalize $\rho$ at each step.


#### Neumann (no-flux / reflecting)

$$
(\beta^{-1}\nabla\rho+\rho\nabla V)\cdot n = -J(\rho)\cdot n = 0
\quad\text{on}\:\:\partial\Omega
\tag{BC-NF}
$$

Particles are reflected, strictly conserving total mass. 
This ensures probability is conserved (no leakage) and is physically natural for a closed domain.
The condition $(\nabla \rho + \rho\nabla V)\cdot n=0$ is nonlinear in $\rho$ and $V$, which may complicate discretization.
(The condition can be simplified by making $V$ flat near boundaries or imposing $\nabla\rho\cdot n=0$ as an approximation.)


### Initial conditions

Choose a probability density $\rho(x,0)=\rho_0(x)$ such that $\rho_0\ge0$ and integrates to 1.

We can consider a localized non-equilibrium Gaussian distribution localized at $x_0$:
$$\rho(x, 0) = \left( \frac{1}{2\pi \sigma^2} \right)^{d/2} \exp\left( -\frac{\|x - x_0\|^2}{2\sigma^2} \right)$$



## Analytical Properties

When designing or benchmarking solvers, the following PDE properties are key:

- **Mass conservation and boundary sensitivity.**  With no-flux BC (BC-NF), the PDE is a conservative continuity equation $\partial_t\rho=-\nabla\cdot J$, so total mass is preserved up to numerical error.

- **Positivity and maximum-principle:**  If $\rho_0(x)\ge0$, then $\rho(x,t)\ge0$ for all $t$.
The Smoluchowski equation is a linear parabolic PDE, the operator is hypoellipcic, with diffusion $\beta^{-1}\Delta\rho$ (which regularizes solution).
Positivity should hold for nonnegative initial data, so solver design (e.g.\ monotonicity-preserving discretizations) should respect positivity whenever possible.  

- **Linear parabolic PDE:**  The equation is linear in $\rho$ and second-order parabolic (elliptic in space). Explicit schemes face a CFL condition: $\Delta t$ must be $O(\Delta x^2)$ due to diffusion or even smaller if the drift $\nabla V$ is large. Implicit or semi-implicit time-stepping is often used to avoid extreme time-step restrictions, especially in high dimensions.  

- **Known equilibrium:**  The steady state $\rho_\infty\propto e^{-\beta V}$【14†L8552-L8560】【14†L8598-L8602】 is known analytically. Solvers can exploit this: for instance, one can check that $\rho_\infty$ is indeed stationary under the discretized operator. Methods that discretize in a weighted space (e.g.\ via $u=\rho/\sqrt{\rho_\infty}$) can make the steady state the constant function, improving conditioning.  

- **Reversible (self-adjoint) structure:**  Because the infinitisimal generator $L = -\nabla V \cdot \nabla + \beta^{-1}\Delta$ is symmetric and self-adjoint in the weighted space $L^2(e^{-\beta V})$ and has a Dirichlet form, one can transform the Fokker-Planck operator into a Schroedinger-type operator. This supports energy-based approaches and preconditioners. This yields orthogonal eigenfunctions and can be used to accelerate convergence (e.g.\ via spectral expansions in known bases for simple $V$). (Pavliotis) This reduction (4.9) for studying Poincaré inequalities【28†L0-L4】, and it implies detailed balance holds【31†L9091-L9094】.

- **Lyapunov functional:**  The free-energy (relative entropy) $\int \rho(\ln\rho + \beta V)\,dx$ is non-increasing in time. In particular, $\rho$ tends to concentrate in the minima of $V$. For well-behaved $V$ this gives a gradient-flow interpretation, which means one can use variational time-stepping or convex splitting schemes if desired.  

Entropy/relative-entropy decay: Pavliotis discusses exponential decay in relative entropy under a logarithmic Sobolev inequality and highlights that this is a better global metric than a restrictive weighted $L^2$ assumption.

H-Theorem and Entropy Stability: The Smoluchowski equation is a gradient flow where the free energy $F[p]$ acts as a Lyapunov functional: $\frac{dF}{dt} \le 0$. Numerical solvers that preserve this discrete free-energy dissipation are guaranteed to be unconditionally stable. 

Pavliotis connects exponential relaxation to equilibrium to functional inequalities of the Gibbs measure, such as Poincaré inequalities (spectral gap) and logarithmic Sobolev inequalities (relative-entropy decay).

**Metastability and tiny spectral gaps** Multimodal $V$ can yield slow convergence (small $\lambda_1$), and metastable transitions can dominate long-time behavior; Chavanis explicitly frames metastability and derives escape-rate behavior in the stochastic Smoluchowski setting.
In practice, nonconvex $V$ (with multiple wells) leads to *metastability*: the density rapidly approaches local equilibria near wells but transitions between wells can be extremely slow.
This means multi-well potentials will test a solver's ability to capture both fast relaxation and slow modes. 
Pavliotis emphasizes that multimodality of $\pi(x)\propto e^{-V(x)}$ (many local minima of $V$) can make convergence slow, because the first nonzero eigenvalue $\lambda_1$ controls mixing rates and is characterized by a Rayleigh-quotient variational principle.



## Potential Pitfalls and Solver Considerations

- **Dimensionality**: One advantage of the linear Smoluchowski PDE is that it is *uncoupled* in each spatial dimension except through the potential $V$; thus for separable $V(x)=\sum_iV_i(x_i)$ the solution factorizes.  This structure can be exploited (e.g.\ using low-rank decompositions or dimension-wise discretization).  

- **Stiffness:**  If $V(x)$ has steep gradients (deep wells or narrow barriers), the drift term $\rho\nabla V$ can be stiff. Explicit schemes will require very small time steps; implicit schemes lead to large linear solves. High condition numbers arise from large second derivatives $V''$. Adaptive time-stepping or operator splitting (solving diffusion and drift separately) can help.  

- **Metastability and multiple time scales:**  As noted, multi-well $V$ yields *metastability*. That is, $\rho$ may quickly relax within each well but only slowly equilibrate globally.  Numerically, one must run long enough (or use acceleration techniques, e.g.\ quasi-stationary approximations) to see the true equilibrium.  Short-time solvers might appear converged to a local equilibrium.  
- Extreme Stiffness via Kramers' Escape Rate: In bistable potentials (like the double-well), the probability of transitioning from a metastable minimum to the global minimum scales exponentially as $e^{-\Delta F / k_B T}$. In high dimensions, the lifetime of these metastable states becomes extraordinarily long. Your time-stepping algorithm must handle extreme stiffness: resolving ultra-fast local equilibration while integrating over vast timescales to capture rare "tunneling" events.
- Pavliotis connects exponential relaxation to equilibrium to functional inequalities of the Gibbs measure, such as Poincaré inequalities (spectral gap) and logarithmic Sobolev inequalities (relative-entropy decay).
He also emphasizes that multimodality of $\pi(x)\propto e^{-V(x)}$ (many local minima of $V$) can make convergence slow, because the first nonzero eigenvalue $\lambda_1$ controls mixing rates and is characterized by a Rayleigh-quotient variational principle.
For benchmarking, this matters because **multiwell potentials** generate (i) long transient plateaus and (ii) sharp separations of time scales—features that stress time integrators, low-rank adaptivity, and sampling-based solvers.

- **Symmetry and scaling:**  If $V(x)$ is isotropic (e.g.\ depends only on $|x|$) or separable, solvers should exploit this to reduce complexity.  Conversely, anisotropic or highly oscillatory $V$ will test the uniformity of the mesh or basis.  Also, remember to set the diffusion coefficient (implicitly $\beta^{-1}$) according to the temperature: low temperature (large $\beta$) makes diffusion small, increasing stiffness and metastability.  

- **Convection-Dominated Regimes**: For steep potentials or large $\beta$ (low temperature), the drift term $\nabla \cdot (\beta p \nabla V)$ heavily dominates the diffusion term. Mesh-based solvers will exhibit severe spurious oscillations (Gibbs phenomenon) unless sophisticated upwinding or stabilization techniques are implemented.

- **Catastrophic Underflow**: The steady-state distribution $e^{-\beta V(x)}$ decays exponentially fast. In 10 or 15 dimensions, density values away from the potential minimum will quickly drop below the standard `float64` minimum precision threshold, causing floating-point underflow. Algorithms often must be reformulated to solve for the log-density $u(x,t) = \ln \rho(x,t)$ to maintain numerical viability.

- **Exploiting the operator form**

The form of the Smoluchowski operator means one can use efficient solvers for symmetric positive-definite systems after an appropriate change of variables.
The known Gibbs measure allows preconditioning or spectral expansion.

Stationary State Form: Because the PDE can be written as $\nabla \cdot (D e^{-\beta V} \nabla (e^{\beta V} p))$, solvers that discretize this specific operator directly (often called Scharfetter-Gummel or exponentially-fitted schemes) can preserve the exact discrete steady state up to machine precision, entirely avoiding spurious steady-state currents.



Table summarizing the properties to exploit and issues to address:

Properties to exploit:
- linear
- positivity (max-principle)
- self-adjoint (reversible)
- known equilibrium distrib
- mass conversion
- energy based approaches?
- schoedinger type operator?




## Example Potentials and Domains

A natural class of benchmarks in $\mathbb{R}^d$ is separable or multi-well potentials.

---
#### The Harmonic well (mutlidiminesional Ornstein–Uhlenbeck process)

1) **Final PDE:** (E1) on $[-L,L]^d$.  
2) **Potential:**
$$
V_{\text{harm}}(x)=\frac{k}{2}\|x\|^2=\frac{k}{2}\sum_{i=1}^d x_i^2,\quad k>0.
\tag{V-H}
$$
3) **BC/IC:** no-flux (BC-NF); IC e.g. (IC-G) centered away from 0.  
4) **Equilibrium:** $\rho_\infty\propto e^{-\beta k\|x\|^2/2}$ (Gaussian).
5) **Solver properties/pitfalls:** smooth, unimodal, strongly log-concave; typically large spectral gap and fast mixing, making it ideal for verifying order-of-accuracy and mass/positivity preservation before moving to metastable cases.
The PDE is then linear with constant drift, and its spectrum is known. This serves as a basic sanity check.  
Represents the multi-dimensional Ornstein-Uhlenbeck process. The dimensions are entirely uncoupled, and the analytical solution is a Gaussian whose variance $\delta^2(t)$ deterministically relaxes to the equilibrium variance $\delta_{eq}^2 = k_B T / K$.
6) **Diagnostics:** mass error; $L^2(\rho_\infty^{-1})$ decay; relative entropy $H(\rho(t)\mid\rho_\infty)$ for entropy-diminishing schemes.

**Truncation heuristic:** for each coordinate, the equilibrium marginal variance is $\approx(\beta k)^{-1}$; take $L\approx m/\sqrt{\beta k}$ with $m\in[5,8]$ to make boundary probability negligible.



---
#### Anisotropic quadratic (stiff but still unimodal)

2) **Potential:**
$$
V_{\text{aniso}}(x)=\frac12\sum_{i=1}^d k_i x_i^2,\qquad 
k_i=k_{\min}\,\kappa^{\,i-1},
\tag{V-AQ}
$$
with $\kappa>1$ (e.g. $\kappa=3$ or $10$) to create strong stiffness across coordinates.  
5) **Pitfalls:** large condition-number in the drift Jacobian and strong scale separation in equilibrium widths; explicit time stepping becomes constrained by the largest curvature direction unless treated implicitly or preconditioned (e.g., via the $h$-transform (E9)).

**Truncation heuristic:** choose coordinate-wise $L_i\approx m/\sqrt{\beta k_i}$ and embed in a box with $L=\max_i L_i$, or better use an anisotropic box if your solver supports it.


---
#### Separable double-well (multimodal; metastability without coupling)

2) **Potential:**
$$
V_{\text{DW-sep}}(x)=\sum_{i=1}^d \frac{1}{4}(x_i^2-a^2)^2.
\tag{V-DWS}
$$
yielding $2^d$ equivalent wells - intoducing metastability.
The resulting equilibrium $\rho_\infty\propto e^{-V}$ has $2^d$ separate peaks, which is challenging for solvers to resolve.

3) **BC/IC:** no-flux; IC as a Gaussian near one well, e.g. $x_0=(a,\dots,a)$ or mixed signs.  
5) **Solver pitfalls:** multimodality $\Rightarrow$ slow equilibration at large $\beta$ (small spectral gap) and very long time horizons to observe inter-well mass transfer; diffusion must be resolved enough to preserve positivity near saddle regions.

**Truncation heuristic:** wells are near $\pm a$; take $L\in[2.5\,a,4\,a]$ depending on $\beta$ (larger $\beta$ concentrates more, so smaller $L$ can suffice).


---
#### Coupled (double-well) Ginzburg–Landau Potential (correlated, non-separable; harder high-$d$)

This is the most useful "next step" beyond separable benchmarks: it creates nontrivial correlations but retains structured coupling.

2) **Potential (nearest-neighbor coupling):**
$$
V_{\text{DW-cpl}}(x)=\sum_{i=1}^d \frac{1}{4}(x_i^2-a^2)^2+\frac{\gamma}{2}\sum_{i=1}^{d-1}(x_{i+1}-x_i)^2,\qquad \gamma\ge 0.
\tag{V-DWC}
$$
5) **Solver properties/pitfalls:** The coupling parameter $\gamma$ destroys separability. This breaks naive tensor-product assumptions and forces the solver to navigate complex, non-separable $d$-dimensional probability landscapes.
Still gradient drift (reversible), but correlations destroy separability; tensor formats become sensitive to variable ordering and rank adaptivity, and particle methods may require many samples to resolve correlated modes. This is conceptually aligned with high-dimensional discretized Ginzburg–Landau–type models used in recent high-$d$ Fokker–Planck work.

**Domain/IC:** $L\approx 3$ is usually safe; IC as Gaussian around one “phase” (all $+1$ or all $-1$).

#### Variant: Coupled quadratic
$V(x)=\tfrac12 x^T A x$ with a positive-definite matrix $A$.  In coordinates this can be $V(x)=\sum_i \alpha_i x_i^2 + \sum_{i\ne j} \gamma_{ij}x_ix_j$.  This still has a single-well, but anisotropy or off-diagonal coupling tests the solver’s handling of mixed derivatives in the drift.  


---
#### Periodic / multiscale potential (oscillatory drift; stresses resolution and aliasing)

Use periodic BCs on $[0,1]^d$ (recommended if your solver supports Fourier/spectral).  

2) **Potential (example with two scales):**
$$
V_{\text{per}}(x)=\sum_{i=1}^d\Big(a\cos(2\pi x_i)+\varepsilon \cos(2\pi x_i/\delta)\Big),
\quad x_i\in[0,1],
\tag{V-P}
$$
with $0<\delta\ll 1$ (fast scale) and $\varepsilon$ moderate.  
5) **Pitfalls:** high-frequency drift $\nabla V$ forces fine spatial resolution (or specialized multiscale representations), and naive discretizations can introduce spurious diffusion/dispersion; excellent for benchmarking spectral accuracy and anti-aliasing strategies.

#### Variant: Periodic or multi-scale:
$V(x)=\sum_{i=1}^d \cos(k_i x_i)$ or sums of such terms produce oscillatory drift.  These test resolution and high-frequency behavior.  


---
#### Rastrigin-type (many local minima; rugged energy landscape)

This is a classic “many-minima” test function adapted as a potential.

2) **Potential:**
$$
V_{\text{Rast}}(x)=A d+\sum_{i=1}^d\left(x_i^2-A\cos(2\pi x_i)\right),\qquad A>0.
\tag{V-R}
$$
5) **Pitfalls:** extreme multimodality $\Rightarrow$ severe metastability at large $\beta$, very small spectral gap, and strong sensitivity to boundary truncation because minima repeat; use it to stress solvers intended for rugged posteriors or energy landscapes.

**Truncation heuristic:** choose $L$ to include several periods (e.g. $L\in[3,6]$).


---
#### Rosenbrock (narrow curved valley; anisotropic stiffness with nonlinearity)

2) **Potential (standard Rosenbrock chain):**
$$
V_{\text{Ros}}(x)=\sum_{i=1}^{d-1}\Big(100(x_{i+1}-x_i^2)^2+(1-x_i)^2\Big).
\tag{V-Ros}
$$
5) **Pitfalls:** essentially unimodal but with a narrow, curved valley—this stresses anisotropic adaptivity, preconditioning, and accurate drift discretization (large $|\nabla V|$ in transverse directions).  

#### Variant" Rosenbrock-like or tilted wells:
Adding linear terms to a multi-well potential (e.g.\ $V(x_i)=(x_i^2-a^2)^2 + \epsilon\,x_i$) breaks symmetry and creates nonuniform minima depth, testing the solver on asymmetric landscapes.  


---
---
### “Difficulty table” for quick benchmark selection

The table below summarizes how each potential tends to stress deterministic solvers (grid/sparse/tensor/spectral) and particle or learning-based solvers.

| Potential | Multimodality | Stiffness in $\nabla V$ | Separability | Typical solver pain point | Suggested $L$ heuristic |
|---|---|---|---|---|---|
| $V_{\text{harm}}$ | low | moderate | separable | baseline verification | $L\sim m/\sqrt{\beta k}$ |
| $V_{\text{aniso}}$ | low | high | separable | time-step stiffness, conditioning | $L_i\sim m/\sqrt{\beta k_i}$ |
| $V_{\text{DW-sep}}$ | high | moderate | separable | metastability, long times | $L\in[2.5,4]$ |
| $V_{\text{DW-cpl}}$ | high | moderate–high | non-separable (local coupling) | rank growth, correlation structure | $L\approx 3$ |
| $V_{\text{per}}$ | medium–high | high (oscillatory) | separable | resolution/aliasing | periodic domain $[0,1]^d$ |
| $V_{\text{Rast}}$ | very high | medium | separable | tiny spectral gap, many wells | include multiple periods |
| $V_{\text{Ros}}$ | low | very high | non-separable | narrow valley resolution | $L$ so valley fits, often $[ -2,2]$ in low-$d$ |


For each potential one typically truncates $\mathbb{R}^d$ to a hypercube $[-L,L]^d$ (with $L$ large enough that $e^{-\beta V(\pm L,\dots)}\approx0$) and prescribes boundary conditions as below. 





## Practical high-dimensional solver families and when they align with Smoluchowski structure

**Low-rank tensor / tensor-network approaches.** Recent open-access work uses tensor trains, cross approximation, and spectral differentiation to reduce degrees of freedom for multidimensional Fokker–Planck equations.
A complementary modern direction represents the solution in a functional hierarchical tensor format and reconstructs $\rho(t,\cdot)$ via density estimation over particles sampled from the associated SDE, explicitly targeting hundreds of dimensions and emphasizing domain truncation choices.

**Sparse grids / Smolyak-type combination strategies.** Sparse-grid philosophy is to combine low-dimensional approximations into a high-dimensional one with improved scaling, and open-access approximation theory continues to expand this toolbox (e.g., kernel-based multilevel + sparse grids).
For Smoluchowski benchmarks, sparse grids are most effective when the solution has mixed regularity and moderate anisotropy rather than extreme multimodality.

**Particle methods / SDE-driven density reconstruction.** Because (E1) is the forward equation of (E2), Monte Carlo and interacting-particle ideas are natural especially in high $d$, and can be paired with density estimation (as above).

**Neural and physics-informed surrogates with tensor structure.** A 2025 paper applies tensor neural networks to steady-state high-dimensional Fokker–Planck equations, explicitly leveraging tensor products of one-dimensional networks to mitigate the curse of dimensionality.
A 2023 paper proposes a low-rank separation representation where $d$ one-dimensional subnetworks assemble a $d$-dimensional solution, re-expressing PDE operators in separable form to avoid exponential growth of training data.

### Mermaid flowchart: choosing and tuning a solver for $d=5,7,10,15$

```mermaid
flowchart TD
  A[Choose potential V and beta] --> B{V separable or nearly separable?}
  B -- yes --> C{Need deterministic grid-based accuracy?}
  C -- yes --> D[Low-rank tensor (TT/Tucker/FTT) + operator splitting\nUse divergence form + h=ρ/ρ∞]
  C -- no --> E[Low-rank separable NN / tensor NN surrogate\nEnforce mass and positivity constraints]
  B -- no --> F{Coupling local (nearest-neighbor) or dense?}
  F -- local --> G[Hierarchical tensor network / TT with good ordering\nPrecondition via h-transform]
  F -- dense --> H[Particle SDE simulation + density estimation\n(or neural operator / flow-based density model)]
  D --> I[Diagnostics: mass, positivity, entropy, L2 decay]
  E --> I
  G --> I
  H --> I
  I --> J{Metastability detected? (slow mixing / small gap)}
  J -- yes --> K[Increase beta-aware resolution / rank / sampling\nUse rare-event-aware diagnostics]
  J -- no --> L[Proceed to harder V or higher d]
```



## Ready-to-Run PDE Formulations

In practice, to plug into a solver one writes the PDE component-wise. For example, setting $\beta=1$ and using Einstein summation, one gets: 

Concretely, one might implement (in $\Omega=[-L,L]^d$): 

- Potential: e.g.\ $V(x) = \frac12|x|^2$ (Gaussian test), or $V(x)=\sum_i (x_i^2-a^2)^2/4$ (double wells), or any of the examples above.  
- BC: Impose $\rho=0$ at $|x_i|=L$ (for all $i$) or $(\nabla\rho+\rho\nabla V)\cdot n=0$.  

By selecting $d=5,7,10,15$, one can then launch a PDE solver.  Key diagnostics are: mass conservation over time, approach to the analytic equilibrium $\rho_\infty\propto e^{-V}$ (e.g.\ compute $\|\rho(\cdot,t)-\rho_\infty\|$), and the ability to capture sharp features of $\rho$. 

The Smoluchowski equation and its properties are discussed in depth in Pavliotis’ *Stochastic Processes and Applications*【14†L8575-L8581】【14†L8586-L8593】, which derives the PDE from the overdamped SDE and analyzes its invariant measures.
Exponential convergence under confining $V$ is also established【31†L9081-L9084】.  (We have not cited every numerical detail in literature, but the above facts follow from those sources and standard Fokker–Planck theory.)