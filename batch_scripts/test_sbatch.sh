#!/bin/bash

#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o slurm-%j.out

python -u ../model.py --test "distilbert-short"