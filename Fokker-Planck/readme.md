


$$ 
\partial_t\,p
+ \nabla\cdot\Bigl(p\,\frac{\nabla U}{\xi}\Bigr)
- D\,\Delta p = 0
$$ 

$$ 
\partial_t\,p
+ \frac{1}{\xi}\nabla p \cdot \nabla U + \frac{1}{\xi}p\,\Delta U
- D\,\Delta p = 0
$$ 

$$ 
\partial_t\,p
- D\,\Delta p
+ \frac{1}{\xi}\nabla p \cdot \nabla U + \frac{1}{\xi}p\,\Delta U
= 0
$$ 

Consider backward formulation (for Feynman-Kac) - just change the sign of $\partial_t\,p$:
$$ 
\partial_t\,p
+ D\,\Delta p
- \frac{1}{\xi}\nabla p \cdot \nabla U - \frac{1}{\xi}p\,\Delta U
= 0
$$ 

We thus have
$$
\sigma = \sqrt{2D} \\
\mu = -\frac{1}{\xi}\nabla U \\
V = \frac{1}{\xi}\Delta U
$$

This gives the SDE
$$
d\mathbf{X}_t = \mathbf{\mu}\, dt + \sigma\, d\mathbf{W}_t \\
d\mathbf{X}_t = -\frac{1}{\xi}\nabla U dt + \sqrt{2D}\,d\mathbf{W}_t
$$



I have a molecule with 7 atoms. The molecule can take on many initial spatial configuraitons corresponding to different positions of the atoms.
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
where \xi and D are just some constants
I have 7 atoms and each has 3 spacial coordinates - I thus have a PDE in 21 dimensional space.
What are some ways of obtaining a solution here?
I do not necesseraly need the value p in the entire domain - but I would like to explore its value in some regions and time intervals.
I am interested in some method that I can employ on any diffusion like high dimensional PDE.
