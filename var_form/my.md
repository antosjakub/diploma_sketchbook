


## Heat

### v0 - original

$$
\frac{\rho_{n+1} - \rho_n}{\delta t} = \alpha \Delta \rho_{n+1} \\

S = \int \frac{1}{2 \,\delta t} (\rho_{n+1} - \rho_n)^2 dx + \int \frac\alpha2 |\nabla \rho_{n+1}|^2 dx \\

\delta S = \int \left( \frac{1}{\delta t} (\rho_{n+1}-\rho_n) - \alpha \Delta \rho_{n+1}\right)\delta\rho \,dx \\

\Delta S = 0 \iff \text{Heat eq PDE above holds}

$$




### v1 - using: $\rho_{n+1} \to \rho_{n+1}+\delta\rho$

$$
S(\rho_{n+1}+\delta\rho)
= \int \frac{1}{2 \,\delta t} (\rho_{n+1}+\delta\rho - \rho_n)^2 dx
+ \int \frac\alpha2 |\nabla \rho_{n+1} + \nabla\delta\rho|^2 dx \\

= \int \frac{1}{2 \,\delta t} \left(
\rho_{n+1}^2 + (\delta\rho)^2 + \rho_n^2
+ 2\,\rho_{n+1}\delta\rho - 2\,\rho_{n+1}\rho_n - 2\,\delta\rho\,\rho_n
\right) dx\\

+ \int \frac\alpha2\left( |\nabla \rho_{n+1}|^2
+ |\nabla\delta\rho|^2
+ 2\,\nabla\rho_{n+1}\cdot\nabla\delta\rho
\right)dx \\

= S(\rho_{n+1}) + \delta S + O(\delta\rho)^2 \\


\delta S = 

\int \frac{1}{\delta t} \left(
\rho_{n+1} - \rho_n
\right)\delta\rho\, dx

+ \int \alpha\left(
 \nabla\rho_{n+1}\cdot\nabla\delta\rho
\right)dx \\


\int_\Omega\nabla\rho_{n+1}\cdot\nabla\delta\rho\,dx
= \int_{\partial\Omega} \nabla\rho_{n+1}\cdot \vec n\,\delta\rho\,dS
- \int_\Omega \Delta\rho_{n+1}\,\delta\rho\,dx \\


\delta S = 
\int \left(\frac{1}{\delta t}
\left(\rho_{n+1} - \rho_n\right)
- \alpha\Delta\rho_{n+1}
\right) \delta\rho\,dx \\
$$



### v2 - using: $\rho_{n+1} \to \rho_{n+1}+\epsilon\phi$


$$
\frac{\rho_{n+1} - \rho_n}{\delta t} = \alpha \Delta \rho_{n+1} \\

S(\rho_{n+1}) = \int \frac{1}{2 \,\delta t} (\rho_{n+1} - \rho_n)^2 dx + \int \frac\alpha2 |\nabla \rho_{n+1}|^2 dx \\


S(\rho_{n+1}+\epsilon\,\phi) = \int \frac{1}{2 \,\delta t} (\rho_{n+1}+\epsilon\,\phi - \rho_n)^2 dx + \int \frac\alpha2 |\nabla \rho_{n+1} + \epsilon\,\nabla\phi|^2 dx \\

\frac{\delta S}{\delta\epsilon}
= \int \frac{1}{\delta t} (\rho_{n+1}+\epsilon\,\phi - \rho_n)\phi\, dx + \int \alpha (\nabla \rho_{n+1} + \epsilon\,\nabla\phi)\cdot\nabla\phi\, dx \\

\frac{\delta S}{\delta\epsilon}\bigg|_{\epsilon=0}
= \int \frac{1}{\delta t} (\rho_{n+1} - \rho_n)\phi\, dx
+ \int \alpha \, \nabla \rho_{n+1}\cdot\nabla\phi\, dx \\


= \int \left( \frac{1}{\delta t} (\rho_{n+1} - \rho_n)
- \alpha \, \Delta \rho_{n+1} \right)\phi\, dx \\

$$

