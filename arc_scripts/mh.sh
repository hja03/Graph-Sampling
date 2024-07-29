#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --partition=short


module load Python/3.11.3-GCCcore-12.3.0
source ../venv/bin/activate
pip list
python mh.py
deactivate
