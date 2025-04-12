#!/bin/bash

find whole_results -mindepth 1 -maxdepth 1 -type d > subfolders.txt

while read subfolder; do
  echo "Enter: $subfolder"

  find "$subfolder" -type f -name "*.sh" | while read shfile; do
    sbatch "$shfile"
    echo "Submitted: $shfile"
  done

done < subfolders.txt
