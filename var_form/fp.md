

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
perform discretization in time (backward euler), so that instead of $\tilde\rho = \tilde\rho(t,x)$, we have $\tilde\rho_n = \tilde\rho_n(x)$
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


### slightly more formal treatment

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


## application in PINNs

### 1a) sequence of N spatial networks

- discretize temporal domain $\{t_n\}_{n=0}^N,\,t_0 = 0, t_N = T\,.$

- train a sequence of neural networks $\{f_n\}_{n=1}^N$
    - $f_n:\Omega \to \mathbb R$
    - $f_n\approx\tilde\rho_n,$


start with the IC $\tilde\rho_0$ and train the network $f_{1}$ by using $S(\tilde\rho_0)[f_1]$ as the loss function, then use network $f_1$ to train $f_2$ via $S(f_1)[f_2]$ etc.


### 1b) single spatial network with N outputs
(pretty much same as 1a but instead of a sequence of networks with spatial input we have a single network with spatial input with a sequence of outputs)

- discretize temporal domain $\{t_n\}_{n=0}^N,\,t_0 = 0, t_N = T\,.$

- train a single network $\vec f(x) = (f_1(x),\dots,f_N(x))$ sequantially
    - $\vec f:\Omega\to\mathbb R^N$
    - $f_n\approx\tilde\rho_n$



start with IC $\tilde\rho_0$ and train only the first output of the network to approximate the minimizer at $t_1$: $f_1(x)\approx\tilde\rho_1$ by using $S(\tilde\rho_0)[f_1]$ as the loss function (ignore the rest $N-1$ network outputs), then use network component $f_1$ to train network output $f_2$ via $S(f_1)[f_2]$ (again, ignore the rest of the network outputs)


### 1c) single spatio-temporal network
(pretty much same as 1a but instead of a sequence of networks with spatial input we have a single network with spatial temporal input)

- discretize temporal domain $\{t_n\}_{n=0}^N,\,t_0 = 0, t_N = T\,.$

- train a single network $f(t,x)$ sequantially
    - $f:(0,T)\times\Omega\to\mathbb R$

start with IC $\tilde\rho_0$ and train the network to approximate the minimizer at $t_1$: $f(x,t_1)\approx\tilde\rho_1$ by using $S(\tilde\rho_0)[f(t_1,\cdot)]$ as the loss function, then use network $f$ now trained at $t_1$ to train it at $t_2$ via $S(f(t_1,\cdot))[f(t_2,\cdot)]$


### 2) just sum all $S$ together


- discretize temporal domain $\{t_n\}_{n=0}^N,\,t_0 = 0, t_N = T\,.$

- train a single network $\vec f(x) = (f_1(x),\dots,f_N(x))$ sequantially
    - $\vec f:\Omega\to\mathbb R^N$
    - $f_n\approx\tilde\rho_n$

- sample $x\sim\frac{\exp{(-\beta V)}}{z}$

- evaluate $S_{tot}$ as the loss function

$$
S_{tot} = \sum_{n=1}^{N-1} S(f_{n+1})

=
\sum_{n=1}^{N-1} S(f_{n+1})
\sum_{x\sim\frac{e^{-\beta V}}{z}} 
\left(
\frac{1}{2\,\delta t}
\left(f_{n+1}-f_n\right)^2
+\frac{\alpha}{2\,\beta} |\nabla f_{n+1}|^2 \\
\right)
$$



### 3) operator learning


We are basically solving the following problem:

given a function $v$, minimize the functional

$$
S(v)[u]
= \int_\Omega
\left(
\frac{1}{2\,\delta t}
\left( u-v\right)^2
+\frac{\alpha}{2\,\beta} |\nabla u|^2 \\
\right)w
$$
where $w$ is some fixed probability distribution defined on $\Omega$.


We can thus train a network to solve this problem for general $v$ - which is basically operator learning problem.

The learned network will then be an operator on some space of functions, $X(\Omega)$:

$f:X(\Omega)\to X(\Omega)$



