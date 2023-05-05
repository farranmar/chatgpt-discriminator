#!/bin/bash

#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00

script_path=$(realpath $0)
script_dir_path=$(dirname $script_path)
proj_dir_path = $(dirname $script_dir_path)
generate="$proj_dir_path/data/generate.py"
echo "script_path=$script_path"
echo "script_dir_path=$script_dir_path"
echo "proj_dir_path=$proj_dir_path"
echo "going to call python ../$generate"
python ../$generate