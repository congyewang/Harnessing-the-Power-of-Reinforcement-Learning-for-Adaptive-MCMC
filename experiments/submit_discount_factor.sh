#!/bin/bash

cd whole_results_discount_factor

for dir in */; do
    echo "Enter: $dir"
    cd "$dir" || continue

    for script in *.sh; do
        if [ -f "$script" ]; then
            sbatch "$script"
            echo "Submitted: $script"
        fi
    done

    cd ..
done
