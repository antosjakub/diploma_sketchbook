# Score-Based Physics-Informed Neural Networks (Score-PINNs)
### Method Summary for High-Dimensional Fokker–Planck Equations

**Source:** :contentReference[oaicite:0]{index=0}

---

## 1. Problem Setting and Motivation

We consider stochastic differential equations (SDEs) of the form

\[
dx = f(x,t)\,dt + G(x,t)\,dw_t
\]

whose probability density \( p_t(x) \) evolves according to the **Fokker–Planck (FP) equation**, a high-dimensional parabolic PDE.

### Core difficulty

In high dimensions:

- **PDF-based methods fail numerically**
  - \( p_t(x) \sim \exp(-\|x\|^2) \) decays exponentially with dimension
  - Leads to **underflow / precision issues**
- **Log-likelihood (LL) PINNs are unstable**
  - Transforming FP → LL yields a **nonlinear HJB equation**
  - PINNs struggle with this nonlinearity and error scales badly with dimension
- **Monte Carlo**
  - Accurate but **slow and sample-inefficient**

---

## 2. Key Idea: Learn the Score Instead of the Density

Define the **score function**

\[
s_t(x) := \nabla_x \log p_t(x)
\]

### Why the score?

This is the central conceptual shift.

#### (1) Numerical stability
- Score does **not decay exponentially**
- Example (Gaussian):
  \[
  s(x) = -\Sigma^{-1}(x - \mu)
  \]
  → linear, dimension-independent scaling

#### (2) Simpler structure
- PDF: complicated + normalization constant
- LL: nonlinear PDE
- Score: often **low-complexity function**

#### (3) Avoids normalization
- Gradient removes partition function

#### (4) Sufficient statistic
From the score, one can recover:
- Log-likelihood (LL)
- PDF
- Sample trajectories

---

## 3. Score-Based Reformulation of the Problem

The paper replaces **“solve FP for \( p_t \)”** with:

> **Step 1:** Learn \( s_t(x) \)  
> **Step 2:** Recover \( q_t(x) = \log p_t(x) \)

---

## 4. Fundamental Equations

### 4.1 Log-Likelihood ODE

Once the score is known, the log-density evolves via:

\[
\partial_t q_t(x) = \mathcal{L}[s_t(x)]
\]

where

\[
\mathcal{L}\[s\] =
\frac{1}{2} \nabla \cdot (GG^T s)
+ \frac{1}{2} \|G^T s\|^2
- \langle A, s \rangle
- \nabla \cdot A
\]

with

\[
A = f - \frac{1}{2}\nabla \cdot (GG^T)
\]

👉 **Important insight:**
- This is an **ODE in time**, not a PDE in space-time
- Spatial complexity is handled by the learned score

---

### 4.2 Deterministic Sampling ODE

The SDE can be replaced by:

\[
dx = \left( A(x,t) - \frac{1}{2} GG^T s_t(x) \right) dt
\]

👉 This gives:
- Deterministic flow
- Faster sampling than Euler–Maruyama

---

## 5. Score-PINN: Learning the Score via a PDE

Instead of matching samples (SM), the paper introduces **Score-PINN**:

> Learn \( s_t(x) \) by solving a PDE it satisfies.

---

### 5.1 Score PDE

The score satisfies:

\[
\partial_t s_t(x) = \nabla_x \left( \mathcal{L}[s_t(x)] \right)
\]

with initial condition:

\[
s_0(x) = \nabla_x \log p_0(x)
\]

Key properties:

- **Second-order PDE in space**
- Derived from FP via differentiation
- Encodes **full physics of distribution evolution**

---

### 5.2 Neural Parameterization

We model:

\[
s_t(x; \theta) \approx s_t(x)
\]

with a neural network.

---

### 5.3 Loss Function (Score-PINN)

\[
\mathcal{L}_{\text{Score-PINN}} =
\lambda_{\text{init}} \, \mathbb{E}_{x \sim p_0}
\left[
\| s_0(x;\theta) - \nabla \log p_0(x) \|^2
\right]
+
\lambda_{\text{res}} \, \mathbb{E}_{t,x}
\left[
\| \partial_t s - \nabla \mathcal{L}[s] \|^2
\right]
\]

### Interpretation

- First term: enforces **initial condition**
- Second term: enforces **Score PDE residual**

---

### 5.4 Computational Considerations

Score-PINN requires:

- First derivatives: \( \nabla s \)
- Second derivatives: \( \nabla^2 s \)

Thus:

- Expensive in high dimensions
- Uses:
  - **Hutchinson trace estimator (HTE)**
  - **Stochastic dimension sampling**

---

## 6. Two-Stage Score-Based SDE Solver

The full method:

---

### Stage 1: Learn the score

Using Score-PINN:

\[
\theta^* = \arg\min \mathcal{L}_{\text{Score-PINN}}
\]

---

### Stage 2: Recover log-likelihood

Solve:

\[
\partial_t q_t(x) = \mathcal{L}[s_t(x;\theta^*)]
\]

via:

- ODE solver or PINN

Loss:

\[
\mathcal{L}_{\text{LL}} =
\| q_0 - \log p_0 \|^2
+
\| \partial_t q - \mathcal{L}[s] \|^2
\]

---

### Important Design Choice

- **Separate networks**:
  - Score model \( s(x,t;\theta) \)
  - LL model \( q(x,t;\phi) \)

Reason:
- Score determines LL
- No mutual regularization needed

---

## 7. Why Score-PINN Works Better than Vanilla PINN

### Vanilla PINN approach
- Learn \( q_t(x) \) directly
- Leads to **nonlinear HJB equation**
- Poor scaling in high dimensions

---

### Score-PINN approach

| Aspect | Vanilla PINN | Score-PINN |
|------|-------------|-----------|
| Target | \( q \) | \( \nabla q \) |
| Equation | Nonlinear HJB | Score PDE |
| Stability | Poor | Good |
| Scaling | Degrades with \( d \) | Stable |
| Structure | Global | Local (gradient-based) |

---

## 8. Conceptual Interpretation

### Score as fundamental variable

The method suggests:

> The score is a more **natural representation of distributions** than density or log-density.

---

### Connection to physics

- Score resembles:
  - **Force field**
  - **Drift correction term**
- Evolution becomes:
  - Deterministic flow driven by learned field

---

### Relation to generative models

- Similar to diffusion models:
  - Learn score
  - Use it to evolve samples
- But here:
  - Forward problem (physics)
  - Not inverse generation

---

## 9. Summary of the Method

### Core pipeline

1. **Reformulate FP problem**
   - From \( p_t(x) \) → \( s_t(x) \)

2. **Learn score via Score-PINN**
   - Solve Score PDE with neural network

3. **Recover log-density**
   - Solve LL ODE

4. **Optional: sampling**
   - Use deterministic ODE

---

### Key advantages

- Avoids:
  - PDF underflow
  - HJB nonlinearity
- Provides:
  - Stable high-dimensional scaling
  - Efficient sampling
- Leverages:
  - Gradient structure of distributions

---

## 10. Practical Takeaways for Implementation

If you already have a PINN codebase:

### Replace:
- Scalar output \( q(x,t) \)

### With:
- Vector output \( s(x,t) \in \mathbb{R}^d \)

---

### Add:

1. **Score PDE residual**
   - Requires:
     - \( \partial_t s \)
     - \( \nabla s \)
     - divergence terms

2. **Initial condition**
   - From known \( p_0(x) \)

3. **(Optional) HTE**
   - For trace terms

---

### Then:

- Solve LL ODE with a second network (or post-process)

---

## Final Insight

The paper’s main contribution is not just a new PINN variant, but a **change of variable**:

> **Density → Score**

This transforms:
- An unstable, high-dimensional PDE problem  
into  
- A stable, learnable vector field problem

which is the key reason **Score-PINNs scale where standard PINNs fail**.

---