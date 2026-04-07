"""
Visualize the LJ gradient and Laplacian for 2 atoms on a 1D line.
Atom 0 is fixed at the origin; the x-axis is the distance r between the atoms.
"""

import torch
import matplotlib.pyplot as plt
from pde_models import FokkerPlanckLJ

# --- model parameters ---
r0      = 1.0
epsilon = 1.0
x_bar   = 10.0   # length scale used for normalisation

model = FokkerPlanckLJ(n_atoms=2, dof_per_atom=1,
                       r0=r0, epsilon=epsilon, L=x_bar)

# --- distance grid (physical units) ---
r_min_plot = 1.0     # just above the divergence
r_max_plot = 4.0
N          = 500

r = torch.linspace(r_min_plot, r_max_plot, N)  # physical positions of atom 1

# Build normalised input Y: atom 0 at 0, atom 1 at r/x_bar
Y = torch.zeros(N, 2)
Y[:, 0] = 0.0            # atom 0: always at origin
Y[:, 1] = r / x_bar      # atom 1: normalised position

grad_flat, laplace_flat = model.precompute_lj(Y)

# precompute_lj returns (B, n_atoms*dof_per_atom)
# index 0 → atom 0, index 1 → atom 1
grad_atom1    = grad_flat[:, 1].detach().numpy()
laplace_atom1 = laplace_flat[:, 1].detach().numpy()

r_np = r.numpy()

# LJ potential: V(r) = 4*epsilon*((r0/r)^12 - (r0/r)^6)
import numpy as np
lj_potential = 4 * epsilon * ((r0 / r_np)**12 - (r0 / r_np)**6)

# LJ equilibrium distance
r_eq = 2 ** (1/6) * r0

# --- plot ---
fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

ax = axes[0]
ax.plot(r_np, lj_potential, color='tab:green')
ax.axhline(0, color='k', linewidth=0.7, linestyle='--')
ax.axvline(r_eq, color='gray', linewidth=0.8, linestyle=':', label=f'$r_{{eq}}={r_eq:.3f}$')
ax.set_ylabel(r'$V_{\rm LJ}(r)$')
ax.set_title('LJ potential')
#ax.set_ylim(-1.5, 3.0)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(r_np, grad_atom1, color='tab:blue')
ax.axhline(0, color='k', linewidth=0.7, linestyle='--')
ax.axvline(r_eq, color='gray', linewidth=0.8, linestyle=':', label=f'$r_{{eq}}={r_eq:.3f}$')
ax.set_ylabel(r'$\partial_x V_{\rm LJ}$')
ax.set_title('LJ gradient (first order 1d derivative)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(r_np, laplace_atom1, color='tab:orange')
ax.axhline(0, color='k', linewidth=0.7, linestyle='--')
ax.axvline(r_eq, color='gray', linewidth=0.8, linestyle=':', label=f'$r_{{eq}}={r_eq:.3f}$')
ax.set_ylabel(r'$\partial_{xx} V_{\rm LJ}$')
ax.set_xlabel('$r$ [$\AA$] (interatomic distance )')
ax.set_title('LJ laplace (second order 1d derivative)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig('lj_2_atoms.png', dpi=150)
print('Saved lj_2_atoms.png')
