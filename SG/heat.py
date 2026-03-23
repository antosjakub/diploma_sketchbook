
###############################################################################
################## SPARSE GRIDS ###############################################
###############################################################################
import pysgpp


# 1. Initialize Grid (Zero Dirichlet boundaries)
dim = 7
level = 5
grid = pysgpp.Grid.createLinearGrid(dim)
grid.getGenerator().regular(level)
gridStorage = grid.getStorage()


# 2. Set Initial Condition u(x,y,0) = sin(pi*x)*sin(pi*y)
import numpy as np
node_values = pysgpp.DataVector(gridStorage.getSize())
for i in range(gridStorage.getSize()):
    pt = gridStorage.getPoint(i)
    node_values[i] = 1.0
    for di in range(dim):
        node_values[i] *= np.sin(np.pi * pt.getStandardCoordinate(di))

alpha = pysgpp.DataVector(node_values)
print(f"len(alpha) = {len(alpha)}")
pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)

# 3. Setup PDE Operators
opLaplace = pysgpp.createOperationLaplace(grid)
laplace_res = pysgpp.DataVector(gridStorage.getSize())

# 4. Time Stepping (Simplified Explicit Euler)
dt = 0.001
steps = 100
diffusivity = 0.1

print("Starting calculation...")
for step in range(steps):
    opLaplace.mult(alpha, laplace_res)
    laplace_res.mult(dt * diffusivity)
    alpha.sub(laplace_res) # Approximation omitting Mass Matrix inversion
print("Calculation done.")



###############################################################################
################## PLOT RESULTS ###############################################
###############################################################################

# --- Resolution of the plot ---
N = 100
x_vals = np.linspace(0, 1, N)
y_vals = np.linspace(0, 1, N)
X, Y = np.meshgrid(x_vals, y_vals)

# --- Evaluate sparse grid surrogate at all meshgrid points ---
op_eval = pysgpp.createOperationEval(grid)
Z = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        pt = pysgpp.DataVector([X[i, j], Y[i, j]] + (dim-2)*[0.23])
        Z[i, j] = op_eval.eval(alpha, pt)

# --- Plot ---
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Heatmap
im = axes[0].pcolormesh(X, Y, Z, cmap='hot', shading='auto')
fig.colorbar(im, ax=axes[0], label='u(x,y,t)')
axes[0].set_title('Heat distribution (pcolormesh)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Contour
contour = axes[1].contourf(X, Y, Z, levels=20, cmap='hot')
fig.colorbar(contour, ax=axes[1], label='u(x,y,t)')
axes[1].set_title('Heat distribution (contourf)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.savefig('heat_solution.png', dpi=150)
plt.close()
#plt.show()
