

"""
PROBLEM
- since we have \tilde\rho, the IC is the hardest, TC is just 1 kinda
- move the majority of the leaning problem to the IC
- do not need to compute AD in the IC problem => training is fast


SETUP
- learn IC first
- init model weights with the prev model's ones

WORKFLOW
- separate training for the IC - just try to match the IC
    - just residual and L2 ?
    - separate args ?
    - separate script
- training loop: model -> model
    - minimization problem


ARGS
- delta t
- training args - forwards them to the main_var.py script


DIFFS

- IC training
    - general training params

- loop training
    - IC trained model to start from
    - delta t
    - minimzalization specific params
    - general training params

needs different Trainer ?
- IC trainer and Var trainer

"""
