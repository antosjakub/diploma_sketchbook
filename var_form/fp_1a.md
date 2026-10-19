

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
starting from
$$
\partial_t\rho
= \nabla\cdot\left(\alpha\rho\nabla V + \frac{\alpha}{\beta}\nabla\rho \right)
$$

substitute
$\rho = \tilde\rho\,e^{-\beta V}$

$$

\partial_t\tilde\rho \,e^{-\beta V}

= \nabla\cdot\left(\alpha\tilde\rho\,e^{-\beta V}\,\nabla V
+ \frac{\alpha}{\beta}\nabla\left(\tilde\rho\,e^{-\beta V}\right)
\right)\\

= \nabla\cdot\left(\alpha\tilde\rho\,e^{-\beta V}\,\nabla V
+ \frac{\alpha}{\beta}e^{-\beta V}\nabla\tilde\rho\,
- \alpha\tilde\rho\,e^{-\beta V}\nabla V
\right)\\

= \nabla\cdot\left(\frac{\alpha}{\beta} e^{-\beta V} \nabla\tilde\rho\right) \\

\partial_t\tilde\rho
= \frac{\alpha}{\beta}\Delta\tilde\rho - \alpha \nabla\tilde\rho\cdot\nabla V
$$


---
perform discretization in time (backward euler), so that instead of $\tilde\rho = \tilde\rho(t,x)$, we work with $\tilde\rho_n = \tilde\rho_n(x)$
$$
\frac{\tilde\rho_{n+1}-\tilde\rho_n}{\delta t} e^{-\beta V}
= \nabla\cdot\left(\frac{\alpha}{\beta} e^{-\beta V} \nabla\tilde\rho_{n+1}\right) \\
$$

and guess the form of the action

$$
S(\tilde\rho_{n+1}) = \int_\Omega \left(
\frac{1}{2\,\delta t}
\left(\tilde\rho_{n+1}-\tilde\rho_n\right)^2 e^{-\beta V}
+\frac{\alpha}{2\,\beta} e^{-\beta V} |\nabla\tilde\rho_{n+1}|^2 \\
\right)
$$

Interpretation
- suppose $\tilde\rho_n$ given
- goal: find $\tilde\rho_{n+1}$ that minimizes $S(\tilde\rho_{n+1})$

Now, lets check the form

$$
S(\tilde\rho_{n+1}+\epsilon\phi) = \int_\Omega \left(
\frac{1}{2\,\delta t}
\left(\tilde\rho_{n+1}+\epsilon\phi-\tilde\rho_n\right)^2 e^{-\beta V}
+\frac{\alpha}{2\,\beta} e^{-\beta V} |\nabla\tilde\rho_{n+1}+\epsilon\nabla\phi|^2 \\
\right) \\

= S(\tilde\rho_{n+1})
+ \delta S
+ O(\epsilon^2) \\


\delta S = \int_\Omega \left(
\frac{1}{\delta t}
\left(\tilde\rho_{n+1}-\tilde\rho_n\right)e^{-\beta V}\epsilon\,\phi
+\epsilon\frac{\alpha}{\beta}\, e^{-\beta V} \nabla\tilde\rho_{n+1}\cdot\nabla\phi \\
\right) \\
= \delta S_I + \delta S_{II}
\\
$$

lets handle $\delta S_{II}$
$$
\delta S_{II} = \epsilon\frac\alpha\beta
\int_\Omega e^{-\beta V} \nabla\tilde\rho_{n+1}\cdot\nabla\phi \\

=
\epsilon\frac\alpha\beta
\int_\Omega \Big(
e^{-\beta V} \nabla\cdot(\nabla\tilde\rho_{n+1}\,\phi)
- e^{-\beta V} \Delta\tilde\rho_{n+1}\phi
\Big)\\

=
\epsilon\frac\alpha\beta
\int_\Omega \bigg(
\nabla\cdot(e^{-\beta V}\nabla\tilde\rho_{n+1}\phi)
+\beta e^{-\beta V}\nabla V\cdot\nabla\tilde\rho_{n+1}\phi
-\,e^{-\beta V}\Delta\tilde\rho_{n+1}\phi
\bigg) \\

=
\epsilon\frac\alpha\beta
\int_\Omega e^{-\beta V} \Big(
\beta \nabla V\cdot\nabla\tilde\rho_{n+1}\phi
-\Delta\tilde\rho_{n+1}\phi
\Big)
+
\epsilon\frac\alpha\beta
\int_\Omega
\nabla\cdot(e^{-\beta V}\nabla\tilde\rho_{n+1}\phi) \\

=
\epsilon\int_\Omega e^{-\beta V} \Big(
\alpha \nabla\tilde\rho_{n+1}\cdot\nabla V
-\frac\alpha\beta
\Delta\tilde\rho_{n+1}
\Big)\phi
+
\epsilon
\int_\Omega
\nabla\cdot\left(\frac\alpha\beta e^{-\beta V}\nabla\tilde\rho_{n+1}\phi\right)
$$

Now lets look at the second integral above. Since the no flux BC conditon is

$$
\vec J\cdot\vec n = 0,\quad 
\vec J =  \left(\frac\alpha\beta e^{-\beta V}\nabla\tilde\rho_{n+1}\right)
$$

then

$$
\int_\Omega \nabla\cdot\left(\frac\alpha\beta e^{-\beta V}\nabla\tilde\rho_{n+1}\phi\right)
= \int_{\partial\Omega} \left(\frac\alpha\beta e^{-\beta V}\nabla\tilde\rho_{n+1}\phi\right) \cdot \vec n = \int_{\partial\Omega} \vec J\cdot \vec n\,\phi = 0
$$

and we get

$$
\delta S_{II} =
\epsilon\int_\Omega e^{-\beta V} \Big(
\alpha \nabla\tilde\rho_{n+1}\cdot\nabla V
-\frac\alpha\beta
\Delta\tilde\rho_{n+1}
\Big)\phi \\


\delta S = \int \left(
\frac{1}{\delta t}
\left(\tilde\rho_{n+1}-\tilde\rho_n\right)

+\Big(
\alpha \nabla\tilde\rho_{n+1}\cdot\nabla V
-\frac\alpha\beta
\Delta\tilde\rho_{n+1}
\Big)
\right)e^{-\beta V}\epsilon\,\phi\,dx \\

\delta S = 0 \iff \text{FP PDE satisfied}
$$


## integral as MC sum

$$
S(\tilde\rho_{n+1}) = \int_\Omega \left(
\frac{1}{2\,\delta t}
\left(\tilde\rho_{n+1}-\tilde\rho_n\right)^2 e^{-\beta V}
+\frac{\alpha}{2\,\beta} e^{-\beta V} |\nabla\tilde\rho_{n+1}|^2 \\
\right) dx \\

= \int_\Omega L_{n+1}\,e^{-\beta V}\,dx \\
= \int_\Omega L_{n+1}\frac{e^{-\beta V}}{z}z\,dx\,, \quad z = \int_\Omega e^{-\beta V}\\

\approx \sum_{\sim\frac{e^{-\beta V}}{z}} L_{n+1}\,z\\

\approx z\sum_{\sim\frac{e^{-\beta V}}{z}} 
\left(
\frac{1}{2\,\delta t}
\left(\tilde\rho_{n+1}-\tilde\rho_n\right)^2
+\frac{\alpha}{2\,\beta} |\nabla\tilde\rho_{n+1}|^2 \\
\right)
$$


## notes

### a) we don't need to compute the value of the partial sum $z$


$z$ appears in two places in the expression for $S$:
1. as constant before the two sums
- Since $z$ is just a constant, it has no impact on the minimalization. We can thus just drop it and minimize $S$ wihout it.
2. in the sampling: $x\sim \frac{e^{-\beta V}}{z}$
- we can sample $x$ using for ex. metropolis, and there, we do not need to know $z$ explicitely



### b) backward euler gives mass / probability conservation:


$$
\frac{\tilde\rho_{n+1}-\tilde\rho_n}{\delta t} e^{-\beta V}
= \nabla\cdot\left(\frac{\alpha}{\beta} e^{-\beta V} \nabla\tilde\rho_{n+1}\right) \\
$$


$$
\int_\Omega \frac{\tilde\rho_{n+1}-\tilde\rho_n}{\delta t} e^{-\beta V}
= \int_\Omega \nabla\cdot\left(\frac{\alpha}{\beta} e^{-\beta V} \nabla\tilde\rho_{n+1}\right)
= \int_{\partial\Omega} \left(\frac{\alpha}{\beta} e^{-\beta V} \nabla\tilde\rho_{n+1}\right)\cdot \vec n \\
= \int_{\partial\Omega} \vec J\cdot \vec n = 0
$$

hence

$$
\int_\Omega \rho_{n+1} = \int_\Omega \rho_{n}
$$


### c) slightly more formal treatment

given $\tilde\rho_n$, minimize the following to obtain $\tilde\rho_{n+1}$

$$
S(\tilde\rho_n)[u]
= \sum_{\sim\frac{e^{-\beta V}}{z}} 
\left(
\frac{1}{2\,\delta t}
\left( u-\tilde\rho_n\right)^2
+\frac{\alpha}{2\,\beta} |\nabla u|^2 \\
\right)
\,,\quad \tilde\rho_{n+1}:=\argmin_u S(\tilde\rho_n)[u]
$$


## application in PINNs: sequence of N spatial networks (1a)

- discretize temporal domain $\{t_n\}_{n=0}^N,\,t_0 = 0, t_N = T\,.$

- train a sequence of neural networks $\{f_n\}_{n=1}^N$
    - $f_n:\Omega \to \mathbb R$
    - $f_n\approx\tilde\rho_n,$


start with the IC $\tilde\rho_0$ and train the network $f_{1}$ by using $S(\tilde\rho_0)[f_1]$ as the loss function, then use network $f_1$ to train $f_2$ via $S(f_1)[f_2]$ etc.


### features

- clean, transparent setup - easy to analyze
- does not need $\lambda_{bc}, \lambda_{ic}, $ etc... - embedded in the formulation
- sampling - just importance from Gibbs. perhaps later add lhs or residual-based - for covering transient regions
- does not need second derivatives
- how do choose $\delta t$ ?
- train, save, train, save, train, etc
- warm start - init the NN with the weights from the prev NN etc - just progressing NNs along trajectory in function space ?
- test: accuracy, mass conservation, comp cost, time duration, scaling with $d$
- experiment with: $\delta t$

### implementation
- accept function (locked NN as $u$)

params:
- input NN filename
- ouput NN filename
- delta t
- network params

kick off an automatic loop that iteratively trains the NNs from $t=0$ to $t=T$ ?





