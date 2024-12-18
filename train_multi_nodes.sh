#!/bin/bash

# please use your dataset path
directory=/scratch/xi9/Large-DATASET/DL3DV-10K/1K
max_jobs=100
l=6000

for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
    Basename=$(basename "$subdir")
    # please use your own path to save data
    check_dir="/scratch/xi9/DATASET/DL3DV-960P-Benchmark-v2/$Basename"
    
    # Avoid to submit too much jobs 
    while true; do
        current_jobs=$(squeue -u $USER -t RUNNING,PENDING | wc -l)
        if [ $((current_jobs-1)) -lt $max_jobs ]; then
            break
        fi
        sleep 5
    done

    if [ -d "$check_dir/lr/24/train_24" ]; then
        echo "Both lr and hr directories exist for $Basename. Skipping job submission."
        continue
    fi
    echo "Both lr and hr directories do not exist for $Basename. sub job submission."
    sbatch train.sh "$directory/$Basename" "$check_dir" $l
    l=$((l + 1))
done