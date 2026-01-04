

Original PINN paper
- https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125
- says that perhaps bayes can be use to tailor the NN architecture to the PDE
- continuous and discrete time models


Convergence of PINN - sequence of minimizers, function spaces etc.
- https://arxiv.org/abs/2004.01806


Expert's guide to PINNS
- https://arxiv.org/abs/2308.08468
- PINN non-dimensionation
- choose fundamental physical units, then scale to unity
- strategies for prioretizing earlier residual via exponential weighting
- 3-6 layers, 128-512 neurons per layer
- 


Large PINN manual, architecture, hyperparams, training, ...
- https://www.sciencedirect.com/science/article/abs/pii/S0952197625020524


Benchmarking models on various PDEs
- https://arxiv.org/abs/2306.08827


Super general PINN pip module
- super good wiki: https://deepwiki.com/lululxvi/deepxde/1-overview
- high level api
- application to various different problems
- and import and use just a snippet from here, ex. gradient calc, ...



Finding the optimal network architecture

PINN-DARTS
- https://arxiv.org/html/2504.11140

Auto-PINN
- https://arxiv.org/html/2205.13748v2/#S4


----

Example PINN code for 1d heat eq.
- pytorch + deepxde grad calc.
- https://dcn.nat.fau.eu/pinns-introductory-code-for-the-heat-equation/




----

Crazy sounding

Reconstructing Relativistic Magnetohydrodynamics with Physics-Informed Neural Networks
- https://arxiv.org/abs/2512.23057

Random Gradient-Free Optimization in Infinite Dimensional Spaces
- https://arxiv.org/abs/2512.20566

PHANTOM: Physics-Aware Adversarial Attacks against Federated Learning-Coordinated EV Charging Management System
- https://arxiv.org/abs/2512.22381


