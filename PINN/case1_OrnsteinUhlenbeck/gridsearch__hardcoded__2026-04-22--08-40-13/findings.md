


## d = 8

n_decay_steps is good
T is good

128 & 256 need a bit more steps


## 4x256 notes

256 almost there, needs a bit a more steps
nice q_if for both d=8 and d=10

256 approx 2x slower training than 128
- score pde 2.5x slower
- ll ode 1.5x slower


256 a bit noticable better in both d=8 and d=10

=> use 4x256 from d=8
- increase num of steps, keep n_decay


4x256 & d=10
up to 6GB ram memory spikes
- grad comp? sampling? backward? - use profiler