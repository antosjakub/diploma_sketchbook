import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

def poisson_matvec(x):
    """Applies the n-dimensional Poisson stiffness matrix to a vector x."""
    x = x.reshape(grid_shape)  # Reshape into n-dimensional grid

    # Apply Laplacian stencil (assuming Dirichlet boundary conditions)
    Ax = np.zeros_like(x)
    Ax += -2 * n * x  # Central coefficient
    for axis in range(n):
        Ax += np.roll(x, shift=1, axis=axis)  # Forward neighbor
        Ax += np.roll(x, shift=-1, axis=axis)  # Backward neighbor

    return Ax.ravel() / h**2  # Flatten result and scale by grid spacing

# Grid parameters
n = 2  # Number of spatial dimensions
grid_size = 10  # Grid points per dimension
grid_shape = (grid_size,) * n  # Shape of the n-D grid
h = 1.0 / (grid_size - 1)  # Grid spacing

# Define LinearOperator
N = np.prod(grid_shape)  # Total number of unknowns
A = LinearOperator((N, N), matvec=poisson_matvec)

# Define right-hand side
b = np.ones(N)  # Example source term

# Solve using Conjugate Gradient
x, info = cg(A, b)

print("Solution:", x.reshape(grid_shape))  # Reshape back to grid
print("Convergence info:", info)

