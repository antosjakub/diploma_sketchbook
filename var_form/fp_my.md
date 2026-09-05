

## Fokker-Planck

consider the PDE
$$
\partial_t\rho =  \nabla\cdot \vec J(\rho) \quad(0,T)\times\Omega\\
\vec J(\rho)\cdot\vec n = 0 \quad (0,T)\times\partial\Omega\\
$$

where
$$
\vec J(\rho) = \alpha\rho\nabla V + \frac{\alpha}{\beta}\nabla\rho

$$


---

$$
\partial_t\rho
= \nabla\cdot\left(\alpha\rho\nabla V + \frac{\alpha}{\beta}\nabla\rho \right)
$$

$$
\rho = \tilde\rho\,e^{-\beta V} \\

\partial_t\tilde\rho \,e^{-\beta V}
= \nabla\cdot\left(\frac{\alpha}{\beta} e^{-\beta V} \nabla\tilde\rho\right) \\

\partial_t\tilde\rho
= \frac{\alpha}{\beta}\Delta\tilde\rho - \alpha \nabla\tilde\rho\cdot\nabla V
$$


---
$$
\partial_t\tilde\rho\,e^{-\beta V}
= \nabla\cdot\left(\frac{\alpha}{\beta} e^{-\beta V} \nabla\tilde\rho\right) \\
$$

guess the form of the action

$$
S(\rho) = \int \left(
\frac{1}{2}
\partial_t \tilde\rho^2 e^{-\beta V}
+\frac{\alpha}{2\,\beta} e^{-\beta V} |\nabla\tilde\rho|^2 \\
\right) dx
$$

now, lets check it

$$
S(\rho+\epsilon\phi) = \int \left(
\frac{1}{2}\partial_t
\left(\tilde\rho+\epsilon\phi\right)^2 e^{-\beta V}
+\frac{\alpha}{2\,\beta} e^{-\beta V} |\nabla\tilde\rho+\epsilon\nabla\phi|^2 \\
\right) dx \\

= S(\rho)
+ \delta S
+ O(\epsilon^2) \\



\delta S = \int \left(
\partial_t
\tilde\rho e^{-\beta V}\epsilon\,\phi
+\frac{\alpha}{\beta}\epsilon\, e^{-\beta V} \nabla\tilde\rho\cdot\nabla\phi \\
\right) dx \\
= \delta S_I + \delta S_{II}
\\
$$


$$
\delta S_{II} = \epsilon\frac\alpha\beta
\int_\Omega e^{-\beta V} \nabla\tilde\rho\cdot\nabla\phi

=
\epsilon\frac\alpha\beta
\int_\Omega e^{-\beta V}

\Big(
\nabla\cdot(\nabla\tilde\rho\,\phi)
- \Delta\tilde\rho\phi
\Big) \\

= 
\epsilon\frac\alpha\beta
\int_\Omega \Big(
\nabla\cdot(e^{-\beta V}\nabla\tilde\rho\phi)
+\beta e^{-\beta V}\nabla V\cdot\nabla\tilde\rho\phi
-\,e^{-\beta V}\Delta\tilde\rho\phi
\Big) \\

=
\epsilon\frac\alpha\beta
\int_\Omega e^{-\beta V} \Big(
\beta \nabla V\cdot\nabla\tilde\rho\phi
-\Delta\tilde\rho\phi
\Big)
+
\epsilon\frac\alpha\beta
\int_\Omega
\nabla\cdot(e^{-\beta V}\nabla\tilde\rho\phi) \\

=
\epsilon\int_\Omega e^{-\beta V} \Big(
\alpha \nabla\tilde\rho\cdot\nabla V
-\frac\alpha\beta
\Delta\tilde\rho
\Big)\phi
+
\epsilon
\int_\Omega
\nabla\cdot\left(\frac\alpha\beta e^{-\beta V}\nabla\tilde\rho\phi\right)
$$

Now lets look at the second integral above. Since the no flux BC conditon is

$$
\vec J\cdot\vec n = 0,\quad 
\vec J =  \left(\frac\alpha\beta e^{-\beta V}\nabla\tilde\rho\right)
$$

then

$$
\int_\Omega \nabla\cdot\left(\frac\alpha\beta e^{-\beta V}\nabla\tilde\rho\phi\right)
= \int_{\partial\Omega} \left(\frac\alpha\beta e^{-\beta V}\nabla\tilde\rho\phi\right) \cdot \vec n = \int_{\partial\Omega} \vec J\cdot \vec n\,\phi = 0
$$

and we get

$$
\delta S_{II} =
\epsilon\int_\Omega e^{-\beta V} \Big(
\alpha \nabla\tilde\rho\cdot\nabla V
-\frac\alpha\beta
\Delta\tilde\rho
\Big)\phi \\


\delta S = \int \left(
\partial_t
\tilde\rho

+\Big(
\alpha \nabla\tilde\rho\cdot\nabla V
-\frac\alpha\beta
\Delta\tilde\rho
\Big)
\right)e^{-\beta V}\epsilon\,\phi\,dx \\

\delta S = 0 \iff \text{FP pde satisfied}
$$

## minimize it

$$
S(\tilde\rho) = \int_\Omega \left(
\frac12\partial_t\tilde\rho^2 e^{-\beta V}
+\frac{\alpha}{2\,\beta} e^{-\beta V} |\nabla\tilde\rho|^2 \\
\right) dx \\

= \int_\Omega L_{n+1}\,e^{-\beta V}\,dx \\
= \int_\Omega L_{n+1}\frac{e^{-\beta V}}{z}z\,dx\,, \quad z = \int_\Omega e^{-\beta V}\\

\approx \sum_{\sim\frac{e^{-\beta V}}{z}} L_{n+1}\,z\\

\approx z\sum_{\sim\frac{e^{-\beta V}}{z}} 


\left(
\frac{1}{2}
\partial_t\tilde\rho^2
+\frac{\alpha}{2\,\beta} |\nabla\tilde\rho|^2 \\
\right)
$$



### minimizing procedure

1. compute / estimate $z$ before the simulation
2. discretize time
3. sample $x\sim\frac{\exp{(-\beta V)}}{z}$
4. evaluate $S$

$$
S = \sum_n S(\tilde\rho) = 

z
\sum_{x\sim\frac{e^{-\beta V}}{z}} 
\sum_n
\left(
\frac{1}{2}
\partial_t\tilde\rho(t_n,x)^2
+\frac{\alpha}{2\,\beta} |\nabla\tilde\rho(t_n,x)|^2 \\
\right)
$$


