#!/bin/bash

cd baseline

for script in *.sh; do
    if [ -f "$script" ]; then
        sbatch "$script"
        echo "Submitted: $script"
    fi
done
