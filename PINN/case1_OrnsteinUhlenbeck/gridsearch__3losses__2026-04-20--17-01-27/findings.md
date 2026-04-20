

full domain sampling + fixed loss weights
    - fine, nice tight decreasing loss
    - might need more iterations - the slope of loss looks nice
    - perhaps do a small grid search over ic loss weight - currently set to 10.0
    - q looks fine but p is deformed near t->T

full domain sampling + adaptive loss weights
    - looking quite
    - the ic is a bit off: the top val for q is -1.62 instead of -1.52
    - q looks fine but is also bit deformed near t->T, better with fixed weights though

trajs sampling
- q not accurate away from origin - not enought grid points...
- with adaptive weights about the same



L = 6 will be allright
combine trajs and full domain sampling?