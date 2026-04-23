
adap weighs with 800 decay
- quite nice
- converges to gaussian
- vect field needs to be a bit better
- same setup but with 128 and more steps??


adap weights 2000 decay much better then 800 step decay!
- try a bit more steps or larger model?



1:20 weights and 800 decay
- keeps the shape from the ic but slightly changes towards gaussian
- is this more accurate to the dynamics or is just fragment of this choice of pde-ic loss weights??



# bottom line
setting 2000 makes the model learn more and not fall into nearest minima

I think adaptive + larger model

