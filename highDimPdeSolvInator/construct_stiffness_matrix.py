from scipy.sparse import kron, eye
from scipy.sparse import diags
import numpy as np

# kernel dies when run with 64**5
dim = 5
N = 64

print(N**dim, int(np.log10(N**dim)))
int_type = np.int8

main_diag = 2 * np.ones(N, dtype=int_type)
off_diag = -1 * np.ones(N-1, dtype=int_type)
L = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(N, N), dtype=int_type)
I = eye(N, dtype=int_type)
A = L
for d in range(1,dim):
    A = kron(I, A) + kron(L, eye(N**d, dtype=int_type))
print(N**dim)
