#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=medium
#SBATCH --clusters=htc
#SBATCH --cpus-per-task=1

echo 'Loading Python'
module load Python/3.11.3-GCCcore-12.3.0

echo 'Activating venv'
source ../venv/bin/activate

echo 'Running Script'
python mh.py --iters 200000 --exponent 500
deactivate

echo ' --- Done --- '
