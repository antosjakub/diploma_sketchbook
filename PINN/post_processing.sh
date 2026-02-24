#!/bin/bash

scripts=()
scripts+=(visualize_training_metrics.py)
scripts+=(visualize_solution_3plots.py)
scripts+=(visualize_solution_3anims.py)


for script_name in "${scripts[@]}"; do
    echo =======================================
    echo Running $script_name ...
    python $script_name
    echo
done