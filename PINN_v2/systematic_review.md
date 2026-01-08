
improve the computational efficiency of PINNs:
- domain decomposition
- parallel computing



high-dim setting?
- dimensional decomposition
- parallel training
- GPU helps to mitigate backprop overhead

PINN
- here, you can play with the architecture more than in the case of traditional solvers
    - diff arch
    - tailor it to the specific problem at hand





## PINN basics

tanh activation
ideal for continuous physical quantity problems due to its excellent smoothness and minimizability. This property enables it to naturally fit complex continuous physical fields and ensures that the output function

automatic differentiation
enables the computation of partial derivatives of the network’s output with respect to input variables time and space directly from any hidden layer

fig 5 - overall NN arch & loss sheme

fig 8 - where can we try to improve smthg in PINNS





## 4.1 Selection of hyperparameters

### adaptive activation funs
sigma(x) -> sigma(a.x), a learnable


### Bayes
Zhang et al. (2019) studied the automatic selection of hyperparameters, especially the learning rate, number of layers, and activation function, through Bayesian optimization. This approach significantly improves the prediction accuracy of the model in uncertain environments and is highly effective in solving parameter uncertainty problems in high-dimensional stochastic PDEs.
https://www.sciencedirect.com/science/article/pii/S0021999119305340?via%3Dihub

### optimizers
use Adam to quickly approximate the optimal solution
followed by L-BFGS for fine-tuning the optimization


### learning rate
decreate over time is good

the warm restart strategy (Loshchilov and Hutter, 2017) has been increasingly applied to the training of PINNs. This approach periodically
resets the learning rate at each stage, enabling the model to escape local
optima and avoid being trapped in suboptimal states

### sampling

in high dims use latin hypercube

see fig 10 for sampling strats visual overview

more sampling in harder regions

sample distrib proportianal to loss fun:
importance sampling method proposed by Nabian
et al. (2021) accelerates the learning process of the model by making
the distribution of sampling points proportional to the loss function.
This method dynamically focuses on regions with larger errors, making
it particularly suitable for solving high-dimensional nonlinear PDE
problems


### loss and residuals

add additional residuals capturing the physics
- symmetries, energy conserv, ...

researchers (Liu and Gerstoft, 2023) have ensured that the model strictly
adheres to specific physical laws by introducing additional physical
constraints, such as conservation laws, temporal invariance, and spatial
smoothing into the loss function. This approach significantly enhances
the accuracy and stability of the model. For instance, Zhang (Zhang
et al., 2023a) incorporated invariance conditions derived from Lie
symmetry into the loss function, improving the model’s ability to
capture physical symmetry and conservation.

The dynamic weight adjustment strategy 
- dynamically balancing the weights of different loss terms during the training process. 
- balanced contribution from each component to the overall optimization
- This prevents certain loss terms from dominating the optimization process or being ignored.
- Such methods effectively address training instability and improve convergence speed in multiscale, multi-objective, and high-dimensional 

- differnet loss function for diff regions
complex geom, turbulence, ...


### auto diff (AD) - strategies

- the computational overhead of AD increases dramatically in high-dimensional and complex models, leading to a significant rise in model training time
- AD is prone to numerical instability when computing higher-order derivatives, especially in challenging scenarios such as eddy currents and high-frequency phenomena 
- AD is less efficient in symbolic differentiation and complex symbolic operations, as the calculation of derivatives of symbolic expressions requires substantial computa

While AD offers high precision in derivative computation, its computational overhead and memory requirements limit its applicability in high-dimensional, complex-structured, and high-order derivative problems.

The computational complexity of AD increases significantly when derivative computations involve
- multi-layer NNs or
- higher-order derivatives

solutions:

Pang et al. (2019)
- proposing fractional calculus PINNs (fPINNs)
- reduce the computational overhead in high-dimensional problems

opPINN framework proposed by Lee et al. (2023)
- enhances the efficiency of symbolic differentiation by introducing symbolic operators to replace certain AD computations
- suitable for complex symbolic computations in highdimensional nonlinear problems


### model arch inovations

LTSM, RNN, CNN, GAN, FNO, ...

DeepONet

CNN accelerates high-dimensional computations

For high-frequency and multi-scale PDEs, Wang et al. (2021c),
based on the theory of neural tangent kernels, utilized Fourier feature
embedding


### domain decomposition

To address the challenge of solving high-dimensional PDEs, Chen et al.
(2024b) proposed the DF-ParPINN method that combines domain decomposition with parallel solving to achieve high efficiency.
By delineating and optimizing locally inefficient regions, this method significantly enhances computational performance in high-dimensional PDEs.

Building on the ideas of spatial and temporal domain decomposition
introduced by Meng et al. (2020a) and Dwivedi et al. (2021), the
DF-ParPINN further integrates parallelization to optimize subdomain
processing. It effectively alleviates the excessive computational burden
associated with traditional PINNs in high-dimensional scenarios.

- https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1309775/full
idea:
decompose domain, and train in parallel


### nonlinear PDEs
weight loss terms based on the domain

### multi scale PDEs
The kernel regression dynamics
of NTK indicate that low frequencies correspond to larger eigenvalues,
causing high-frequency gradients to decay faster during backpropagation, forming a bottleneck where high-frequency features are difficult
to capture (Wang et al., 2022b). This further results in PINN having relatively weak expressive power for multi-frequency problems,

Fourier feature methods can enhance the multi-scale modeling capabilities of NNs by explicitly encoding high-frequency information.
For example, FD-PINN maps physical equations to the frequency domain, using the discrete Fourier transform to convert high-dimensional
multivariate PDEs into low-complexity frequency-domain equation systems, and employing frequency-domain residual constraints to mitigate training instability caused by high-frequency oscillations in the
physical domain (Li et al., 2025a).
- use fourier to transform the eq, then solve in fourier space - smooth
The fundamental breakthrough of such
methods lies in transforming rigid oscillations in the physical domain
into smooth representations in the frequency domain.


### stochastic PDEs
When the system response exhibits stochastic process characteristics due to parameter uncertainty
or random excitation (Yang et al., 2018), traditional deterministic PDEs
fail to characterize probabilistic distribution properties and thus become ineffective. In such cases, stochastic differential equations (SDEs)
must be adopted as the governing equations to quantify uncertainty
as joint probability density functions

- probability distribution constraints (solutions must satisfy complex statistical properties such as non-Gaussianity and multimodality) (E et al., 2017; Raissi, 2018)

The Bayesian inference framework provides a rigorous means of
quantifying uncertainty. Srilatha et al. (2024) combines an adaptive
sampling mechanism with high-order discretization techniques. The
SDE discretization is achieved through adjusted Euler–Maruyama or
Milstein schemes, combined with Markov chain Monte Carlo (MCMC)
variational inference strategies, to achieve rapid convergence of the
parameter posterior distribution in sparse noise data scenarios.


use weak form of the eq??
- sol does not have to be that smooth
- BC are satisfied better




## Limitations
Theoretical studies demonstrate that the generalization error of
PINN possesses a rigorous upper bound (Mishra and Molinaro, 2023),
which is jointly constrained by the training error and the number of
collocation points. This mathematical guarantee indicates that when
the collocation point density sufficiently covers key features within
the solution space, such as shock waves, boundary layers, and other
high-gradient regions, PINN can achieve high-fidelity approximation
within unobserved regions of the training domain through embedded
physical constraints (i.e. PDE residual terms). A

separable Kolmogorov-Arnold NNs for a little bit more dim PDEs
- https://arxiv.org/abs/2411.06286

Notably, despite incorporating
physical constraints, practical PINN training often demands dense collocation points, complex architectures, and extensive epochs to achieve
high-accuracy, physically consistent solutions, resulting in considerable
time expenditure.


While multi-objective optimization provides the flexibility to integrate data and physical laws in PINN design and training,
its inherent challenges can significantly compromise training stability and solution quality. 

Techniques like
- dynamic weight adjustment,
- adaptive learning rates,
- staged optimization strategies
offer promising avenues for mitigating these limitations
Such methodological advances may enhance PINN efficiency and efficacy for increasingly
complex physical systems.

The minimization of the loss function
constitutes a complex, multi-objective optimization problem. While
this function-space-based global approximation offers considerable expressive power, it often proves difficult to ensure the simultaneous,
balanced, and precise satisfaction of the governing equation residual, boundary constraints, and initial conditions during optimization.

- IC/BC propagation failure
However, when the information contained in initial/boundary conditions fails to propagate effectively into the interior via the network, optimization may not converge to the true physical solution.
Propagation failure is characterized by sharply skewed high-residual regions acting as ‘‘barriers’’ obstructing information flow.

quantum computer - good hardware for PINNs?
Quantum parallel computing offers transformative potential for PINNs. Quantum
neural networks (QNNs) leverage quantum superposition and exponential memory capacity to compress parameter dimensionality while accelerating high-dimensional physics-constrained solutions. Preliminary validation emerges from studies like Dehaghani et al. (2024),
where QNN-parameterized quantum states (e.g., density matrices ̃𝜌)
reduce storage complexity from 𝑂(4^n) in classical PINNs to O(n^2)
effectively mitigating the curse of dimensionality in high-dimensional
quantum system simulations.
