
Hi, I am a math / physics masters student and I am interested in numerically solving diffusion-like parabolic PDEs: heat eq, classical diffusion, Fokker-Planck, Schroedinger, … - in higher dimensions - mainly 5,6,7,..,30 (and beyond if possible).
I heard that sparse grids might be useful in this context.


---

I am interested in numerically solving diffusion-like parabolic PDEs: heat eq, classical diffusion, Fokker-Planck, Schroedinger, … - in higher dimensions - mainly 5,6,7,..,30 (and beyond if possible).
I heard that sparse grids might be useful in this context.
How do does the method work?

(You are a math expert. The target audience are master / phd. students who are familiar with math / physics - pdes, finite different, finite elements numerical schemes, interpolation,...)

Write a short conceptual summary using the following publications:
https://en.wikipedia.org/wiki/Sparse_grid
https://cran.r-project.org/web/packages/SparseGrid/vignettes/SparseGrid.pdf
https://ins.uni-bonn.de/media/public/publication-media/sparsegrids_j8NLaMi.pdf?name=sparsegrids.pdf
https://sparsegrids.org/
If you will need some more try to rely on trusted sources - mainly university lecture notes.

Use the attached article on sparse grids (which contains multiple figures) to write a technical chapter explaining how exactly the method works.
Be clear, exact, and detailed here. Include a few figures if they would be helpful.
Use the following document:
https://ins.uni-bonn.de/media/public/publication-media/sparsegrids.pdf?pk=91
The final text should be about 5 pages long.

-- make some changes here --

Now, considering what you learned so far, write a chapter directly focusing on the possible issues in high dimensions and way of resolving them. What issue does / can sg face is higher dimensions? How to resolve them?
Give a note about practical implementation - hardware requitements - GPU / cpu / ram / ...

-- make some changes here --
Now give the example of some heat equation with a non trivial source term in 6d domain and how to apply sg to solve such a system.
Use sg++ to show how would the code look like.
Use mainly the following page: https://sparsegrids.org/


Again, if you will need some more try to find some from trusted sources - mainly university lecture notes.

What should the chapter look like?
- intro - how it works, diagram
- curse of dim - how it adresses it, what problems can arise
- example on some toy model that captures the essence