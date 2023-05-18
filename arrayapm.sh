#!/bin/bash
#PBS -l select=1:ncpus=1
#PBS -l select=1:ncpus=1:mem=90000M
#PBS -l walltime=24:00:00
#PBS -J 1


module add lang/python/anaconda/3.8.8-2021.05-2.5
export PYTHONPATH=$PYTHONPATH:/user/work/aj20377/Auto-PyTorch

# Define executable
#export EXE=/bin/hostname

# Change into working directory
#cd ${PBS_O_WORKDIR}

# Do some stuff
#echo JOB ID: ${PBS_JOBID}
#echo PBS ARRAY ID: ${PBS_ARRAY_INDEX}
#echo Working Directory: $(pwd)

#echo Start Time: $(date)
# Execute code
python3 test2.py
#echo End Time: $(date)

