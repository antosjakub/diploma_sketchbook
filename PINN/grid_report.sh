#!/bin/bash

#dims=("1" "2" "3" "4" "5" "6" "7" "8")
d_list=("1" "2" "3" "4" "5")
layers_list=("64" "64,64" "64,64,64")

## Iterate through all combinations
for layers in "${layers_list[@]}"; do
for d in "${d_list[@]}"; do
    echo ======================================
    echo layers = $layers, d = $d
    python main.py --n_steps=300 --n_steps_log=100 --n_steps_lbfgs=0 --layers=$layers --d=$d --profiler_report_filename=prof_rep_layers=$layers,d=$d
done
done