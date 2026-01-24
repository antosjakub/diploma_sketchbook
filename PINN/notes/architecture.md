use perplexity to collect articles





custom PDE
- bc, id, pde
- bc, id, pde residual

tests
- various different ic, bc
- train network and compare max err against analytic sol

streamlit visualize


t, x1, x2, ...


Boundary value problem
need to define
- pde (pde residual)
- ic
    - ic function
    - ic residual
- bc
    - dirichlet
        - bc funcitonn
        - bc residual
    - neumann
        - bc residual
-> train.py need to only know the residuals