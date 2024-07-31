#!/bin/bash
#SBATCH --job-name=EKI        # Name of the job
#SBATCH --array=0                   # Number of tasks in the array (e.g., 10 jobs)
#SBATCH --partition=expansion
#SBATCH --time=48:00:00                # Maximum runtime for each job (hh:mm:ss)
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=64G           # Memory required per job
#SBATCH --output=outputEKI.out  # This specifies where the stdout will go

# Run your Python script with the GPU device specified (assuming your script is named 'your_script.py')
python EKI_L96s.py
