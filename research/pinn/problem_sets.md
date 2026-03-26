
# Numerical methods for solving diffusion PDEs on high-dimensional domains


We consider a following set of PDEs defined on d-dimensional domains.


## Heat equation

$$
\partial_t u - \alpha \Delta u = 0
$$
with the analytical solution:
$$
u(t,x) = \Bigg(\prod_{i=1}^d \sin{(k_i\,x_i)}\Bigg)\:e^{-\alpha |k|^2 t}
$$


## Heat equation II

$$
\partial_t u - \alpha \Delta u = f
$$
where
$$
f(t,x) = -\beta \Bigg(\prod_{i=1}^d \sin{(k_i\,x_i)}\Bigg)\:e^{-\alpha |k|^2 t} \sin{\beta t}
$$
with the analytical solution:
$$
u(t,x) = \Bigg(\prod_{i=1}^d \sin{(k_i\,x_i)}\Bigg)\:e^{-\alpha |k|^2 t} \cos{\beta t}
$$


## Travelling gaussian packet


Consider the following PDE:

$$
\partial_t u - \delta \Delta u + \vec{v} \cdot \nabla u + w\, u = f
$$

where:
$$
f(x,t) = -
\Bigg[
4\alpha^2\delta \sum a_i^2(a_ix_i-b_i+c_it)^2\,
+ \beta
\Bigg]
e^{-\alpha\sum_{i=1}^d(a_i x_i - b_i + c_i t)^2 -\beta t} \\
v_i = -\frac{c_i}{a_i} \\
w = -2\delta\alpha\sum a_i^2
$$

With the analytic solution in a form of a travelling gaussian packet:
$$
u(x,t) = e^{-\alpha\sum_{i=1}^d(a_i x_i - b_i + c_i t)^2}\,e^{-\beta t}
$$



## Travelling gaussian packet II


Consider the following PDE:

$$
\partial_t u - \delta \Delta u + \vec{v} \cdot \nabla u + w\, u = f
$$

where:
$$
f(x,t) = -
\Bigg[\Bigg\{
4\alpha^2\delta \sum a_i^2(a_ix_i-b_i+c_it)^2\,
+ \beta\Bigg\}
\cos{\gamma t} + \gamma \sin{\gamma t}
\Bigg]
e^{-\alpha\sum_{i=1}^d(a_i x_i - b_i + c_i t)^2 -\beta t} \\
v_i = -\frac{c_i}{a_i} \\
w = -2\delta\alpha\sum a_i^2
$$

With the analytic solution in a form of a travelling and oscillating gaussian packet:
$$
u(x,t) = e^{-\alpha\sum_{i=1}^d(a_i x_i - b_i + c_i t)^2}\,e^{-\beta t}\cos{\gamma t}
$$


## Radially symmetric PDE on unit ball

Consider a ball of radius 1 centered at 0.

$$
\partial_t u - \delta \Delta u  = f
$$

where
$$
f(x,t) = (-\beta + 4\delta |x|^2) \sin(|x|^2 + a t) e^{-\beta t}
$$
and $a=2d\delta$

With the analytic solution
$$
u(x,t) = \sin(|x|^2 + a t) e^{-\beta t} 
$$



## Fokker-Placks: Molecule with 7 atoms
Consider a molecule with 7 atoms. The molecule can take on many initial spatial configuraitons corresponding to different positions of the atoms.
I have an algorhtm that takes in two different configuratiosn of the molecule, computes some paths in the space of configurations, and returns the most probable reaciton time - the time needed to get from one configutation to the other.
I would like to arrive at similar result but from the direction of PDEs on high dimensional domains.
Lagevin equation for the system:
$$
d\mathbf{X}_t = \mathbf{\mu}\, dt + \sigma\, d\mathbf{W}_t \\
d\mathbf{X}_t = -\frac{1}{\xi}\nabla U dt + \sqrt{2D}\,d\mathbf{W}_t
$$
The equation can also be rewritten into a Fokker-Plack PDE with some interatomic potential U:
$$
\partial_t\,p - D\,\Delta p
+ \frac{1}{\xi}\nabla p \cdot \nabla U + \frac{1}{\xi}p\,\Delta U = 0
$$
where \xi and D are just some constants.
I have 7 atoms and each has 3 spacial coordinates - I thus have a PDE in 21 dimensional space.


# A few optional

## Hamilton-Jacobi

to be added


## PDE on manifold (general theory of relativity)

to be added


## Some pde from thermodynamics

to be added