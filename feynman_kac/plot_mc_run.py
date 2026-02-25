
import matplotlib.pyplot as plt
import main
import numpy as np


# Parameters
T = 1.0
dt = 0.01
N_paths = 10^5
# Coordinates of where we want to get the solution
x0 = 0.3
t0 = 0.0


# for plotting
t0 = 0.0
a, b = -1.5, 1.5
nx = 20
x_domain = np.linspace(a,b,nx)

u = np.zeros_like(x_domain)
for i,x0 in enumerate(x_domain):
    u_MC, std_err = main.fk_advection(t0, x0, T, N_paths, dt)
    u[i] = u_MC


plt.plot(x_domain, u)
plt.savefig('plot.png')


