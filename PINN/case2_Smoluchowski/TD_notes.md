



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



### how to confirm it all right?

## IC and last loop TC
- lmax
- l2 err
- rel l2 ?

### every loop:
- loss
- mass conservation

## other
- compare with comb-tech
- exact solver in low dims ?



Test
- accuracy
- mass conservation
- computational cost
- training stability
- first- vs second-order autodiff
- convergence with $\Delta t$
- scaling with dimension.

- interpolation accuracy metric
    - double the n of interpl points, how does the L2 err change
        - possible math theor handling - theorems etc
    - decrease teh time delta, does the accuracy change much ?
    - increase decrease numer of netwrok params

    - activaitons - softplus not tanh

    - julia Voronoi package 

    - in case of MC, presample the points at once, then jsut take batches

    - sample a large pool of 100,000 poitns before training,
        - then, every k-th step, do a few more steps of metropolis on the large pool
        - parallel on gpu


## Thesis building up plan

# Part 1
1. compare agains strong 2d
2. correct
3. compare agains strong 2d
etc.

- both for pinn & comb-tech

# Part 2
scale to higher dimes and look at accuracy metrics:
- mass consrvation
- IC and TC error


-----------------------

Reference sol in lower dims - FEM ?



Combitech
- tried FE, failed
- FVM discret perpahps better ?


