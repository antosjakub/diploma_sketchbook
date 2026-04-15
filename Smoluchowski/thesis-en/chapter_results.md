

we choose $\beta = 1.0$


# Case 1: Diffusive Drift


First, we consider a simple case of a linear uniform drift.
This drift is given by the potential of the form
$$
V(x) = \sum_{i=1}^d c_i x_i
$$
representing a constant force field (ex. a uniform electric field pulling ions). The analytical solution is a drifting and decaying gaussian..




# Case 2: Harmonic potential
- separable...


# Case 3: Coupled Quadratic
- separable...

## Computing the partition function

$$
V(x) = \frac{1}{4}\sum_{i=1}^d (x_i^2-a_i^2)^2
$$

$$
z = \int_{(-\infty,\infty)^d} e^{-\beta V(x_1,...,x_d)} dx_1...dx_d \\
= \prod_{i=1}^d \left( \int_{-\infty}^\infty e^{-\frac{\beta}{4} (x_i^2-a_i^2)^2} dx_i \right)
$$

we uniformly sample $a_i$ from the interval $(0.85,1.15)$

choosing L = 3.5 is enough

using numpy's polyfit, we get
$$
\int_{-\infty}^\infty e^{-\beta (x^2-a^2)^2} dx \approx -1.048 a^3 + 1.618 a^2 + 0.03337 a + 2.438 =: z_{fit}(a)
$$

--show plot--




## Sparse Grids

## vanilla-PINN

## Score-PINN


