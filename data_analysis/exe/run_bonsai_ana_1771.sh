#!/bin/sh

# Activate Environment
source /mnt/lustre/scratch/nlsas/home/usc/ie/dcr/software/python_envs/py39_venv/bin/activate

# Run Initial Data Selection
python3 src/read_and_process_for_bonsai.py --run 1771 --parts 10 --chargeCut True

# Run BONSAI Analysis
python3 src/nicf_bonsai.py --run 1771

