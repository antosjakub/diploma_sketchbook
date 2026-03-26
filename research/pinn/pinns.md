

I am interested in numerically solving diffusion-like parabolic PDEs: heat eq, classical diffusion, Fokker-Planck, Schroedinger, … - in higher dimensions - mainly 5,6,7,..,30 (and beyond if possible). (I do not have any data from experiments available.)
I heard that PINNS might be useful in this context.
How do does the method work?

You are a math expert.

You goal is to write a ~20 page report based on which other coding specific llm agent can than write a high-performace code for high-dim PINNs in JAX.

Use the attached pdfs on PINNS to write the report consisting of the following 5 chapters (in latex):

1) A short conceptual summary - about 1 page long - about PINNS.

2) A technical chapter explaining how exactly the method works in general.
Be clear, exact, and detailed here. Use latex for equations.
The chapter should be about 5 pages long.

3) A technical chapter listing the various different architectures discussed in the attached papers that might be relevant for convergence, accuracy, and efficiency in higher dimensions.
Create a subsection for each one.
About 10 pages long.

4) A chapter discussing the most important strategies / ideas for solving high-dim pdes. About 2 pages.

5) A chapter detailing how exacly to apply PINNs on a concrete pde. Consider some heat equaiton with a diffusion and source term, walk through the setup, and how to code it. Be very detailed. About 3 pages.




Consider the following report on solving PDEs on high-dimensional domains using PINNs: pinn_report.tex.
Use the report and jax resources online to create a high-performace python jax module for solving PDEs on high dimensional domains.
I am specifically interested in using PINNS on the test problems listed in the problem_sets.md file.
I want high-efficiency, high-performace code nicely divided into multiple importable files. Each function should have a small clean in-code documentation. Include a functionality for visualizing the main training results - trained network, loss, l2 error, ... I want to be able to test multiple architectures via grid search. A very minimal memory / time training reporting would also be nice. The main objective is to push it to as many dimensions as possible while maintaining clean code. Prefer minimal simple design over bloat.

This repo (https://github.com/lululxvi/deepxde/tree/master) contains some existing jax code, you can use it for inspiration.

