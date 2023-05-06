# NotChatGPT

welcome to our DL final project :)

<br> 

## Generating data on Oscar
1. SSH into oscar:  
`ssh -X <username>@ssh.ccv.brown.edu`  

<details>
  <summary>Notes for setting up the first time</summary>
  
  The first time you do this, you must clone this repo and install all necessary packages etc. I tihnk the easiest way to do this is by creating the same `csci1470` conda environment we used for all the HWs by following HW0, since it has most of the packages we need except for openai. Once you do this and have activated the environment, install openai:  
    `conda install -c conda-forge openai`  
</details>

<br>

2. Make sure `generate.py` has the correct starting index, number of requests, and number of threads set

3. `cd` into the `batch-scripts` directory  

4. Set your api key environment variable:  
`export APIKEY=<your api key>`

4. Run this command to submit the batch job:  
`sbatch generate_data_sbatch.sh`  
It will return a batch job number, the output of this job will be saved in the `batch-scripts` directory in a file named `slurm-<job number>.out`

5. Once the job finishes running (idk how to tell when it is done tbh, other than checking the slurm file), the generated data will be stored in the various `thread0.csv`, `thread1.csv`, etc files in the `data/threads` directory. To combine these all into one file, `cd` into the `data/threads` directory and make sure all the files in there contain data you want to combine. If there are old files from a previous run that used more threads, delete the ones from old threads. Once the directory contains all the csvs you want to combine, run:  
`cat *.csv >combined.csv`  
Then, move the `combined.csv` file to the `data/partials` directory (and consider renaming it to indicate what indexes of data are in there).

6. Push the generated data to git  


## Training on Oscar

1. SSH into oscar:  
`ssh -X <username>@ssh.ccv.brown.edu`  

<details>
  <summary>Notes for setting up the first time</summary>
  
  The first time you do this, you must clone this repo and install all necessary packages etc. I tihnk the easiest way to do this is by creating the same `csci1470` conda environment we used for all the HWs by following HW0, since it has most of the packages we need except for openai. Once you do this and have activated the environment, install openai:  
    `conda install -c conda-forge openai`  
    I also had to install libstdcxx-ng to train:  
    `conda install -c anaconda libstdcxx-ng`
</details>

<br>





