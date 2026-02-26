#!/bin/bash

d_list=("1" "2" "3" "4" "5" "6" "7" "8" "9")

## Iterate through all combinations
for d in "${d_list[@]}"; do
    echo ======================================
    echo d = $d
    #python main.py --n_steps=300 --n_steps_log=100 --n_steps_lbfgs=0 --d=$d --output_dir_name=run_mem_pde --profiler_report_filename=prof_rep_layers=d=$d
    python main.py --n_steps=300 --n_steps_log=100 --n_steps_lbfgs=0 --d=$d --output_dir_name=run_mem_weak_form --profiler_report_filename=prof_rep_layers=d=$d --use_weak_form
done