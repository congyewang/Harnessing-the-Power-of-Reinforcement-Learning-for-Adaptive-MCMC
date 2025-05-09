#!/bin/bash

cd search_step_size

for script in *.sh; do
    if [ -f "$script" ]; then
        sbatch "$script"
        echo "Submitted: $script"
    fi
done
