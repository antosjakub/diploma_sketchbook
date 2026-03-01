
21-d FP
use SiLu activations
use exp as the final activation for FP - output positive
4-8 layers
64-128 neurons per layer
for rough potential - try modified - residual like connections
10^4 - 10^6 collocation points
normalize the space so that typical distance you care about is O(1)
for point sampling - use the existing Jan algorihm for finding optimal reaction path
two phase training
- 1. focus on IC res
- 2. focus on pde res