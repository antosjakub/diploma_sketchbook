from scipy.sparse import kron, eye
from scipy.sparse import diags
import numpy as np

N = 10

# 1D finite difference stencil [-1, 2, -1]
main_diag = 2 * np.ones(N)
off_diag = -1 * np.ones(N - 1)
T = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(N, N))

# Create 2D Laplacian using Kronecker product
I = eye(N)  # Identity matrix
L = kron(I, T) + kron(T, I)