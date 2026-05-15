#!/bin/sh

# Activate Environment
source /mnt/lustre/scratch/nlsas/home/usc/ie/dcr/software/python_envs/py39_venv/bin/activate

# Run Initial Data Selection
python3 src/read_and_process_for_bonsai_vTc.py --run 1767 --parts 20 --chargeCut True --tc_cut 5.0 --q20_min 500 --q20_max 12500

# Run BONSAI Analysis
python3 src/nicf_bonsai.py --run 1767

