import numpy as np
import matplotlib.pyplot as plt
import json

# Load Z (space-delimited text)
Z_ex = np.loadtxt('slice_data_ex.txt')
Z_fk = np.loadtxt('slice_data_fk.txt')

# Load metadata
with open('slice_data.json', 'r') as f:
    meta = json.load(f)

x1 = np.linspace(meta['x1_range'][0], meta['x1_range'][1], int(meta['x1_range'][2]))
x2 = np.linspace(meta['x2_range'][0], meta['x2_range'][1], int(meta['x2_range'][2]))
X1, X2 = np.meshgrid(x1, x2)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot
im = axes[0].pcolormesh(X1, X2, Z_fk, shading='auto', cmap='viridis')
axes[0].set_xlabel(rf'$x_{meta["varying_dims"][0]}$')
axes[0].set_ylabel(rf'$x_{meta["varying_dims"][1]}$')
axes[0].set_title('Feynmac-Kac MC')
plt.colorbar(im, label='f(x, t)', ax=axes[0])

im = axes[1].pcolormesh(X1, X2, Z_ex, shading='auto', cmap='viridis')
axes[1].set_xlabel(rf'$x_{meta["varying_dims"][0]}$')
axes[1].set_ylabel(rf'$x_{meta["varying_dims"][1]}$')
axes[1].set_title('Analytic')
plt.colorbar(im, label='f(x, t)', ax=axes[1])

plt.savefig('slice_plot.png')
