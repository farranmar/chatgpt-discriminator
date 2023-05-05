#!/bin/bash

#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00


echo "going to call python -u ../data/generate.py"
python -u ../data/generate.py