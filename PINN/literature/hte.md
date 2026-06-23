

https://arxiv.org/pdf/2312.14499


## basics

computes the hessian as it walks the graph

taylor mode AD

does not do backward, does forward


## story

f: \R^d \to \R

backward only needs to be done once, but requires more comp power
forward needs to be done d times....
hence backward is standard..

but if u use taylor AD..

## implementaion

implemented in jax

similar torch version exist as well:

https://github.com/f-dangel/torch-jet/
https://torch-jet.readthedocs.io/en/latest/generated/gallery/01_taylor_mode/#download_links



to benchmark against
https://docs.pytorch.org/functorch/main/generated/functorch.hessian.html



## other I found along the way
https://openreview.net/pdf?id=nl1ZzdHpab

torch pinn implem
https://github.com/rezaakb/pinns-torch/tree/main
