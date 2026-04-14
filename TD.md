

chapters
- pinns - arch
- score pinn
- sparse grids





1
use bc in score pinn

2
use general domain & time dimensions - pass in the sampling object
ex. L_i_min, L_i_max <- tensor of shape (d, 2) & also T

- combine the two trainers into one
- run only pde, ic, bc




score pinn

vanilla pinn
- head:
    - no, exp, softplus

gs:
    - trajs, lhs, mixed traj & lhs
    - bc: neuman, dirichlet, no
    - L = 2.0, 3.0, 4.0









# focus: pinns and smoluchowski

write background to smoluchowski
- get an idea for the requirents on V
- look out for stuff to be useful for PINN, sparse grids, ...

finish intro chapter about pinns / score pinns
- sdgd
- make them readable

smoluchowski & score-pinns
- try
    - harmonic - just k^2/2 |x|^2
    - coupled - lot of diff params here, make it for general 1/2 x^T A x
    - double well - 2 wells better than one