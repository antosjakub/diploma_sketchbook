
# AN EXPERT’S GUIDE TO TRAINING PHYSICS-INFORMED NEURAL NETWORKS
- https://arxiv.org/pdf/2308.08468
- code: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main


General notes on the paper
- gives us a few values to investigate when training the PINN
    - spectral bias - functions of low frequencies first, issues with high freq
        - look at NTK evals
    - causality - the loss favour minimizing the residuals at the later time first
- gives us a few arch to try out and see
- easy to follow the ideas
- sometimes wrongly formatted and typed equations
- the math behind it is not that much rigorous, do not feel confident in the result



## 1. Introduction
numerous studies focusing on improving the performance of PINNs, mostly by:

- designing more effective neural network architectures or better training algorithms.
    - loss re-weighting schemes have emerged as a prominent strategy for promoting a more balanced training process and improved test accuracy
    - other efforts aim to achieve similar goals by adaptively re-sampling collocation points, such as importance sampling [30], evolutionary sampling [31] and residual-based adaptive sampling

- developing new neural network architectures to improve the representation capacity of PINNs.
    - adaptive activation functions [33], positional embbedings [34, 35], and novel architectures [26, 36, 37. 38, 39, 40]

- alternative objective functions for PINNs training, beyond the weighted summation of residuals [41].

- numerical differentiation [42], while others draw inspiration from Finite Element Methods (FEM), adopting variational formulations [43, 44].

- adding additional regularization terms to accelerate training of PINNs

- training strategies, Techniques such as sequential training [47, 48] and transfer learning [49, 50, 51] have shown potential in speeding up the learning process and yielding better predictive accuracy


issues in PINNs training, including
- spectral bias [52, 35],
- unbalanced back-propagated gradients [26, 27]
- causality violation


ingredients for success
- non-dimensionalization
- network architectures that employ
    - Fourier feature embeddings
    - random weight factorization,
- training algorithms
    - causal training,
    - curriculum training
    - loss weighting strategie


## 2. PINNS

critical training pathologies prevent PINNs from yielding accurate and robust results
- spectral bias [52, 35],
- causality violation [53],
- and unbalanced backpropagated gradients among different loss terms [26]
- ...

To address these issues, we propose a training pipeline that integrates key recent advancements, which we believe are indispensable for the successful implementation of PINNs.
Three main steps:
- PDE non-dimensionalization,
- suitable network architectures,
- employing appropriate training algorithms.


## 3. Non-dimensionation

The initialization of neural networks has a crucial role on the effectiveness of gradient descent algorithms

init. schemses can prevent vanishing gradients also accelerate training convergence
A critical assumption for these initialization methods is that input variables should be in a moderate range, ex. zero mean and unit variance, which enables smooth and stable forward and backward propagation

If the variables are not properly scaled, the optimization algorithm may have to take very small steps to adjust the weights for one variable while large steps for another variable. This may result in a slow and unstable training process.


## 4. Network architecture

strats:

- hyper-parameter choices, activation functions, and initialization schemes. 
- random Fourier feature embeddings - enables coordinate MLPs to learn complex high frequency functions
- random weight factorization - a simple drop-in replacement of dense layers, to accelerate training convergence and improve model performance

### 4.1 MLP

- good MLP math into here

layers: 3-6
neurons: 128-512
tanh activations - good second der
init dense layers with Glorot scheme

### 4.2 Random Fourier features

MLS suffer from spectral bias - they are biased towards learning low freq. functions
- issue with learning high freq funs and fine structures

x -> [cos(Bx), sin(Bx)] =: gamma(x)

x.shape = [d,1]
B.shape = [m,d]
gamma.shape = [2m,1]

entries of B sampled from rand Normal distrib N[0,\sigma^2], \sigma\in[1,10]

- for sharp gradients & and sharpness
- for low dims only?

idea comes from here
- https://github.com/tancik/fourier-feature-networks


### 4.3 Random weight factorization

Instead of W^l, we have s^l,V^l
- grad descent with respect to s, V
- init as so:
Glorot scheme on W, normal dist on s -> W = diag(exp(s)) V -> get init of V

params can take on higher range of values
geometric intuition
consistently and robustly improve the loss convergence and model accuracy

see the following for more details (almost the same as in the curr paper):
- https://arxiv.org/pdf/2210.01274


## 5 Training

### 5.1 Temporal causality

conventional PINNs tend to minimize all PDE residuals simultaneously meanwhile they are undesirably biased toward minimizing PDE residuals at later time, even before obtaining the correct solutions for earlier times. 

To impose the missing causal structure... we do (just for pde res loss)

1) split time domain into M parts
2) compute loss in each part
3) compute total loss as weighed mean with weights w_i, i=1,..,M
using losses from prev parts:
w_i = exp( - eps \sum_{k=1}^{i-1} loss_k )

The temporal weights encourage PINNs to the PDE solution progressively along the time axis

sensitive to eps
choose eps=1.0
the larger the eps the more strongly the temporal causality is enforced


### 5.2 Loss balancing

we have 3 losses:

- pde residual
- bc residual
- ic residual

how to combine them? just sum them up? with what weights?

a) grad based scheme
lambda = alpha * lambda_old + (1-alpha) * lambda_new
lambda_new_BC = [sum of all grad losses] / [grad loss of BC]
these updates can either take place every hundred or thousand iterations of the gradient descent loop

b) NTK matrix
- hight comp cost
- recipe:
take the grad of u at each sampled point with respect to NN params and compute average over the NN params
get matrix of size N_sampled_pnts x N_sampled_pnts
compute the trace (= sum of evals, evals of NTK characterize the conv rate of the loss func.)
compute lambda based on that
then moving average
- NTK more stable then grad based
- more comput. demanding


### 5.3 curriculum training

Time marching:
divide time domain into M intervals and solve the PDE in each
- choose IC as the last timestep from the previous interval
- significantly reduces the optimization difficulty of learning the full evolution of a dynamical system while increasing computational costs due to model retraining for each window


Progressive approach
ex. to solve Navier Stokes with high turbulence (high Rey. num), pretrain a model with a smaller Rey num
- it reduces the likelihood of PINNs becoming trapped in unfavorable local minima, thus enabling them to accurately capture complex and nonlinear PDE solutions


## 6 Miscellaneous


### 6.1 Optimizer

Adam is good
- do not use weight decay
- lr - crutial for PINN
    - initial lr of 1e-3
    - exp decay

### 6.2 Random sampling

- use random sampling, do not use full batch sampling
- basically a regularization
Based on our observations, training PINNs using full-batch gradient descent may result in over-fitting the PDE residuals. Consequently, we strongly recommend using random sampling in all PINN simulations to achieve optimal performance.


### 6.3 Imposing BCs

you can enforce them
to enforce space or time periodicity choose a special fourier feature embedding
to enforce dirichlet, neumann, etc. use [65,66] - not easy though


### 6.4 Modified MLP

for learning nonlinear and complex PDE solutions

the only change is in using more complex actiation scheme:
g_l = sigma(f_l)
g_l = sigma(f_l) .dot U + (1-sigma(f_l)) .dot V
U = sigma(W_1 x + b_1)
V = sigma(W_2 x + b_2)

U,V - two encoder networks
the modified MLP demands greater computational resources; however, it generally outperforms the standard MLP in effectively minimizing PDE residuals, thereby yielding more accurate results


## Results

base:

4 layers
256 neurons
tanh
Glorot init scheme
Adam optim
lr
- init_lr 1e-3
- exp decay
- decay rate of 0.9 every 2,000 decay steps
uniform domain sampling
batch size 4096
num iterations = 10,000







Casual: 5.1 Temporal causality: split temporal domain into M parts
RWF: 4.3 Random weight factorization: W_l -> s_l, V_l
Grad Norm: 
Fourier Feature: 


FF - important overall
RWF - sometimes very good
modified MLP - good for non-linear PDEs