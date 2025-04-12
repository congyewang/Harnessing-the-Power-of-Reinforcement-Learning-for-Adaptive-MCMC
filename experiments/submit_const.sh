#!/bin/bash

cd laplace_1/const
for file in *.sh; do
    sbatch "$file"
done
cd -

cd laplace_2/const
for file in *.sh; do
    sbatch "$file"
done
cd -

cd laplace_4/const
for file in *.sh; do
    sbatch "$file"
done
