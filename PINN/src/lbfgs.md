  - PyTorch’s LBFGS expects a closure that reevaluates the same objective during one
    optimizer step, and it is described as a memory-intensive quasi-Newton method with
    repeated function evaluations per step. That setup matches stable, low-noise losses,
    not highly stochastic minibatches.
    Source: PyTorch docs
    https://docs.pytorch.org/docs/2.12/generated/torch.optim.LBFGS.html

  - A recent PINN optimization study compares Adam, L-BFGS, and Adam+L-BFGS, and uses a
    fixed set of 10,000 residual points plus fixed boundary/initial points for all 41,000
    iterations in its experiments. It reports Adam+L-BFGS as the strongest of the three.
    Source: Rathore et al., 2024
    https://arxiv.org/abs/2402.01868

  - In general optimization, standard L-BFGS is known to prefer full-batch / low-noise
    gradients; changing the sample every iteration can make quasi-Newton updates unstable.
    Sources:
    https://arxiv.org/abs/1802.05374
    https://arxiv.org/abs/1707.08552

  - In PINNs specifically, there is a separate line of work on adaptive resampling /
    refining the training set during training, but that is typically done as an outer
    procedure: train on a set, then refresh/refine the set, then continue.
    Sources:
    DeepXDE / RAR overview: https://arxiv.org/abs/1907.04502
    DAS-PINNs: https://arxiv.org/abs/2112.14038
    R3 sampling: https://arxiv.org/abs/2207.02338