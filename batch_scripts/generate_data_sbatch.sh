#!/bin/bash

#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00


echo "going to call python -u ../data/generate.py"
python -u ../data/generate.py --api_key $APIKEY