
## 4x64

freq 100
    - n_steps decay 200 - converges perhaps too quickly
    - n_steps decay 2000 - loss will still decrease with longer training 
    - both ll_ode losses go up & down quite a lot

freq 1000
    - segments of with 1000 steps
    - visible segment with high loss, then segment with low loss
    - does not go that super much up and down though


4x64 & decay 200: for freq 100 and 1000 almost exact same q_T
4x64 & decay 2000: for freq 100 and 1000 almost exact same q_T
=> freq makes difference in how loss looks like but not in final model

all 4 pretty much the same
p_T ~ max 0.0085, reference is 0.006128
q_T ~ max -4.7 min -22, reference max -5.05 min -18


## 4x256

res freq 100
- needs less decays - loss convergers too quickly
- needs more decays - loss goes down but oscillates quite a bit

# Bottom line

4x64 or 4x256 does not matter

n_steps decay 2000 reach lower losses than 200, but oscillate more

n_steps decay of 400-1000 might be ideal

nice loss: layers=256_256_256_256__resampling_frequency=100__n_steps_decay=200
- maybe do not decrease the learning rate every 200 steps but every 400?

all give the same q_T


## next steps:

try 4x64 on higher dims with 400-1000 n_decay steps

try longer T, T = 1.0, 3.0 to get a different q_T