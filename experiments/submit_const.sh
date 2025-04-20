#!/bin/bash

# Define the list of directories to be processed
directories=("banana/const" "neals_funnel/const" "laplace_1/const" "laplace_2/const" "laplace_4/const")

# Traverse each directory
for dir in "${directories[@]}"; do
    echo "Processing directory: $dir"
    cd "$dir" || continue # Skip if the directory does not exist

    # Submit all sh files in the directory
    for file in *.sh; do
        [ -f "$file" ] && sbatch "$file" # Ensure the file exists before submitting
    done

    cd - >/dev/null # Return to the original directory, suppressing output
done
