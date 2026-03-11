
# Numerical methods for solving diffusion PDEs on high-dimensional domains


We consider a following set of PDEs defined on d-dimensional domains.


## product of sines

$$
\partial_t u + \alpha \Delta u = 0
$$
with the analytical solution:
$$
u(t,x) = \Bigg(\prod_{i=1}^d \sin{(a_i\,x_i)}\Bigg)\:e^{-\alpha |a|^2 t}
$$



## Gauss diffusion

$$
\partial_t u + a \nabla u \cdot v + b \Delta u = 0
$$
where a,b are constant and v is a constant vector
and the IC reads:
$$
u(t=0,x) = \frac{1}{(2\pi)^{d/2}} e^{-0.5|x|^2}
$$




## Travelling gaussian packet

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
