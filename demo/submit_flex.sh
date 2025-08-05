#!/bin/bash

cd gmm_2d_40

for script in *.sh; do
    if [ -f "$script" ]; then
        sbatch "$script"
        echo "Submitted: $script"
    fi
done
