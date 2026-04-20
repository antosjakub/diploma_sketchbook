

full domain sampling & fixed loss weights = super trash: fishtank with liquid on top right
- moving to adaptive weights fixes this
    - for some reason super high bc loss
    - nice tight decreating loss

full trajs sampling & fixed loss weights = super trash: q super strange, p explodes to 1e7
- moving to adaptive weights fixes this
    - for some reason super high bc loss
    - decreasing loss with a lot of variance



## takeaways:
- the box width can be smaller ~ L=8
- ignore bc, do just pde,ic
- use adaptive weights
- both traj and domain may work here, domain sampling give much better results here

- even adaptive weights + full domain sampling works relatively good with pde,bc,ic
    - just need smaller L box to be more accurate near the origin
    - perhaps pair it with some traj sampling?