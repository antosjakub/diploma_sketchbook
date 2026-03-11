import numpy as np
import matplotlib.pyplot as plt
import json

# Load Z (space-delimited text)
Z = np.loadtxt('slice_data.txt')

# Load metadata
with open('slice_data.json', 'r') as f:
    meta = json.load(f)

x1 = np.linspace(meta['x1_range'][0], meta['x1_range'][1], int(meta['x1_range'][2]))
x2 = np.linspace(meta['x2_range'][0], meta['x2_range'][1], int(meta['x2_range'][2]))
X1, X2 = np.meshgrid(x1, x2)

# Plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(X1, X2, Z, shading='auto', cmap='viridis')
plt.colorbar(label='f(x, t)')
plt.xlabel(rf'$x_{meta["varying_dims"][0]}$')
plt.ylabel(rf'$x_{meta["varying_dims"][1]}$')
plt.title('2D Heatmap Slice')
plt.savefig('slice_plot.png')
