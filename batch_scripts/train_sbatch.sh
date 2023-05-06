#!/bin/bash

#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

script_path=$(realpath $0)
script_dir_path=$(dirname $script_path)
proj_dir_path=$(dirname $script_dir_path)
model="$dir_path/model.py"
echo "going to call python $model --save_weights --num_epochs 1"
python $model --save_weights --num_epochs 1