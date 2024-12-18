#!/bin/bash
#SBATCH --job-name 3DGS
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 64gb
#SBATCH --time 12:00:00
#SBATCH --gpus a100:1

source /etc/profile.d/modules.sh
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate data_generator

python train_render.py --scene_path $1 --output_path $2 --use_lr --port $3