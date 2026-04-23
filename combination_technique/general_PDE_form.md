

# General problem


We are interested in numerically solving d-dim FOkker-Planck PDEs:

$$
\partial_t p = \alpha \Delta p - \nabla \cdot (\mu p)
$$

and possibly also

$$
\partial_t p = div(A\nabla p) - \nabla \cdot (\mu p)
$$

where A is a constant full rank matrix 


on a general [-L,L]^d domain, with either neumann or dirichlet BC and usually some gaussian or other IC
