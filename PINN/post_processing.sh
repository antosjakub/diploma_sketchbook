#!/bin/bash

if [ $# -eq 1 ]; then
    dir_name="$1"
elif [ $# -eq 0 ]; then
    dir_name="run_latest"
else
    echo "Error: Exactly one argument expected, but $# provided. Usage: $0 <dir_name>" >&2
    exit 1
fi

echo "Using directory: $dir_name"

scripts=()
scripts+=(visualize_training_metrics.py)
scripts+=(visualize_solution_3plots.py)
scripts+=(visualize_solution_3anims.py)


for script_name in "${scripts[@]}"; do
    echo =======================================
    echo Running $script_name $dir_name ...
    python $script_name $dir_name
    echo
done