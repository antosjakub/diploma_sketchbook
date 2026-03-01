#!/bin/bash

## Define the options for each argument
n_steps_lbfgs=0
n_steps=15_000
n_steps_log=200

n_steps_decay=500

bs_options=("16")
e_options=("25")
N_options=("5" "10" "15")
k_options=("2" "4" "8")
sd_options=("0.0" "0.2" "0.4")

# best result:
# --batch_size=16 --epochs=25 --N=5 --k=8 --stochastic_depth=0.0


## Iterate through all combinations
for bs in "${bs_options[@]}"; do
    for e in "${e_options[@]}"; do
        for N in "${N_options[@]}"; do
            for k in "${k_options[@]}"; do
                for sd in "${sd_options[@]}"; do
                    echo --batch_size=$bs --epochs=$e --N=$N --k=$k --stochastic_depth=$sd
                    #python 3d_recognition.py --batch_size=$bs --epochs=$e --N=$N --k=$k --stochastic_depth=$sd
                    /home/antosjak/deep_learning/venvDL/bin/python3.11 3d_recognition.py --batch_size=$bs --epochs=$e --N=$N --k=$k --stochastic_depth=$sd
                done
            done
        done
    done
done