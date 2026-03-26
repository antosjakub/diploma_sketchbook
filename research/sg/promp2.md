

I am interested in numerically solving diffusion-like parabolic PDEs: heat eq, classical diffusion, Fokker-Planck, Schroedinger, … - in higher dimensions - mainly 5,6,7,..,30 (and beyond if possible).
I heard that sparse grids might be useful in this context.
How do does the method work?

(You are a math expert. The target audience are master / phd. students who are familiar with math / physics - pdes, finite different, finite elements numerical schemes, interpolation,...)


Use the attached pdfs on sparse grids (which contains multiple figures) to write the following 3 chapters (in latex):

1) A short conceptual summary - about half a page long.

2) A technical chapter explaining how exactly the method works.
Be clear, exact, and detailed here. Use latex for equations.
The chapter should be about 15 pages long.
Do not list every lemma, but explain in detail the core concepts.
Include a few figures and an algorithm outlining the method.
Have a special subsection about the energy norm.

3) A chapter detailing how exacly to apply the sg on a concrete pde. Consider some heat equaiton with a diffusion and source term, walk through how to discretize and construct the system, and how to solve it. Be very detailed. About 4 pages.




Use the attached article on sparse grids (which contains multiple figures) to write a technical chapter explaining how exactly the method works.
Use the following document:
https://ins.uni-bonn.de/media/public/publication-media/sparsegrids.pdf?pk=91
Be clear, exact, and detailed here. Use latex for equations.
The final text should be about 6 pages long.
Do not list every lemma, but explain in detail the core concepts.
Include a few figures and an algorithm outlining the method.

Then write a short chapter ~ 1 page directly focusing on the possible issues in high dimensions and way of resolving them. What issues does SG face is higher dimensions? How to resolve them?




Now give the example of some heat equation with a non trivial source term in 6d domain and how to apply sg to solve such a system.
Use sg++ to show how would the code look like.
Use mainly the following page: https://sparsegrids.org/

Give a note about practical implementation - hardware requitements - GPU / cpu / ram / ...
code

Again, if you will need some more try to find some from trusted sources - mainly university lecture notes.

What should the chapter look like?
- intro - how it works, diagram
- curse of dim - how it adresses it, what problems can arise
- example on some toy model that captures the essence