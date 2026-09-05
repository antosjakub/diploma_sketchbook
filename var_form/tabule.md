



## tabule 1.9.2026

### Laplace

$$
0 = \Delta \rho \\

S(\rho) = \int \frac12|\nabla \rho |^2 \,dx \\

S(\rho+\delta\rho)
= \int \left( \frac12|\nabla \rho |^2 + \nabla \rho \cdot \nabla \delta \rho + O(\delta\rho)^2\right) dx \\

= S(\rho) - \int \Delta\rho\:\delta\rho \, dx + O(\delta\rho)^2 \\

= S(\rho) + \delta S + O(\delta\rho)^2 \\

\delta S = - \int \Delta\rho \,\delta\rho \, dx = 0 \iff \Delta \rho = 0 \\


$$


### Heat

$$
\frac{\rho_{n+1} - \rho_n}{\delta t} = \alpha \Delta \rho_{n+1} \\

S = \int \frac{1}{2 \,\delta t} (\rho_{n+1} - \rho_n)^2 dx + \int \alpha |\nabla \rho_{n+1}|^2 dx \\

\delta S = \int \left( \frac{1}{\delta t} (\rho_{n+1}-\rho_n) - \alpha \Delta \rho_{n+1}\right)\delta\rho \,dx \\

\Delta S = 0 \iff \text{Heat eq PDE above holds}

$$



### FP

$F$ - free energy

$$
F = \int \left(\frac{1}{\beta} \rho \ln\rho + V\rho\right)dx \\

\frac{\delta F}{\delta \rho} = \frac{1}{\beta}\left(\ln\rho+1\right) + V \\

\Xi = \int \frac12 d\rho (\nabla\rho^*)^2 dx,\quad
\rho^* = \frac{\delta F}{\delta\rho}
$$

$$
\partial_t\rho
= \frac{\delta\Xi}{\delta\rho^*}\bigg|_{\rho^*}
= -\nabla\cdot\left(\alpha\rho\nabla\rho^*\right)\bigg|_{\rho^*}
= -\nabla\cdot\left(\alpha\rho\nabla V + \frac{\alpha}{\beta}\nabla\rho \right)
$$

then subsitute
$$
\rho = \tilde\rho\,e^{-\beta V} \\

\partial_t\tilde\rho \,e^{-\beta V}
= -\nabla\cdot\left(\frac{\alpha}{\beta} e^{-\beta V} \nabla\tilde\rho\right) \\

\partial_t\tilde\rho
= -\frac{\alpha}{\beta}\Delta\tilde\rho + \alpha \nabla\tilde\rho\cdot\nabla V
$$

min functional then

$$
S
= \int \left(\frac{e^{-\beta V}}{2\,\delta t}\left(\tilde\rho_{n+1} - \tilde\rho_n\right)^2 + \frac{\alpha}{2\,\beta} e^{-\beta V} |\nabla\tilde\rho_{n+1}|^2\right) dx \\

S \approx \sum_{x\sim e^{-\beta V}} S_{n+1},
\quad S_{n+1} = 
\frac{1}{2\,\delta t}\left(\tilde\rho_{n+1} - \tilde\rho_n\right)^2 + \frac{\alpha}{2\,\beta} |\nabla\tilde\rho_{n+1}|^2
$$



