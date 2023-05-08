#!/bin/bash

#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out

python -u ../data/generate-from-title.py