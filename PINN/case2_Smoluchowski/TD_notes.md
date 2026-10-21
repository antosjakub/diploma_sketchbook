



## Best so far:

```
python main_var_IC.py --n_steps=9900 --gamma=0.999
```
- with uniform sampling on max domain 2.15


- n_steps=10_000
- gamma=0.999
very good choice this, perhaps needs a bit more steps to converge
lr:
- lr_start: 0.001
- lr_end:   0.000781



ALSO good:
```
python main_var_IC.py --n_steps=50000 --gamma=0.99 --n_steps_decay=200
```
lr:
- lr_start: 0.001
- lr_end:   0.000811

sampling
- uniform on domain 2.15
- gaussian filter < 2.0
- metropolis filter < 2.0
- uniform on domain 2.15, gaussian filter < 2.0

uniform is good overall
gaussian gives visibly good results at the center of the IC


moving from 50k to  100k steps makes the err 2-4x smaller (plot_variational_step.png)



## TD


### sampling
- currenlty only uniform on fixed interval [-2.15, +2.15]

- combine: Gaussian + Metropolis
- bind Metropolis only to some regions so that it does not leave the Gibbs pot shape that often
    - perhaps constrain it by contour ?
    - just let it evolve, then filter by

- Metropolis sampling takes a long time, perhaps let it run in parallel on cpu ?

