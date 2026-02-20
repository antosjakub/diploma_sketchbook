
# Heat equation with a source term

$u_t - \alpha\, \Delta u = f \:\:\text{on}\:\:\Omega$

initial & boundary conditions:

$u(t=0, x) = u_0(x) \:\:\text{on}\:\:\Omega$

$u(t, x) = u_{D}(t,x) \:\:\text{on}\:\:\partial\Omega$



## Discretization in 1 dimension

$\partial u / \partial x \approx \frac{u(x+h/2) - u(x-h/2)}{h}$

$\partial^2 u /\partial x^2 \approx \frac{u(x+h) - 2u(x) + u(x-h)}{h^2}
\approx \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2}$

$\partial u / \partial t \approx \frac{u(x+\tau) - u(x)}{\tau}
\approx \frac{u^{t+1} - u^{t}}{\tau}$

### a) explicit

$$
\frac{u_i^{t+1} - u_i^{t}}{\tau}
- \alpha\, \frac{u_{i+1}^{t} - 2u_i^{t} + u_{i-1}^{t}}{h^2}
= f_i^t
$$

$$
u_i^{t+1}
= \frac{\alpha\,\tau}{h^2} (u_{i+1}^{t} - 2u_i^{t} + u_{i-1}^{t}) + u_i^{t} + \tau f_i^t
$$

$$
U^{t+1}
= \frac{\alpha\,\tau}{h^2} L\,U^t + U^{t} + \tau F^t
= (I + \frac{\alpha\,\tau}{h^2}L) \, U^{t} + \tau F^t
$$

We just iteratively compute $U^{t+1}$ according to:

$$
U^{t+1} = (I + \frac{\alpha\,\tau}{h^2}L) \, U^{t} + \tau F^t
$$

where $U_0$ is given by the initial condition.

### b) implicit

$$
\frac{u_i^{t+1} - u_i^{t}}{\tau}
- \alpha\, \frac{u_{i+1}^{t+1} - 2u_i^{t+1} + u_{i-1}^{t+1}}{h^2} = f_i^{t+1}
$$

$$
u_i^{t+1} -
\frac{\alpha\,\tau}{h^2} (u_{i+1}^{t+1} - 2u_i^{t+1} + u_{i-1}^{t+1})
= u_i^{t} + \tau f_i^{t+1}
$$

$$
U^{t+1} -
\frac{\alpha\,\tau}{h^2} L\,U^{t+1}
= (I - \frac{\alpha\,\tau}{h^2} L)\, U^{t+1}
= U^t + \tau F^{t+1}
$$

To get $U^{t+1}$ from $U^{t}$, we need to solve:

$$
(I - \frac{\alpha\,\tau}{h^2} L)\, U^{t+1} = U^t + \tau F^{t+1}
$$

(We need to solve $A\,x=b$, where $x$ and $b$ are of size $n^d$.)

- $A = (I - \frac{\alpha\,\tau}{h^2} L)$
- $x = U^{t+1}$
- $b = U^t + \tau F^{t+1}$