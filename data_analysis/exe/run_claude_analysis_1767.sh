#!/bin/sh

# Activate Environment
source /mnt/lustre/scratch/nlsas/home/usc/ie/dcr/software/python_envs/py39_venv/bin/activate

# Run Analysis
python run_analysis.py --sig-run 1767 --bkg-run 1766 --n-parts 1 --thresh-min 2 --source-pos 0 1525 0 --data-dir /mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/raw_data/production_v0  --geo-json /home/usc/ie/dcr/hk/ambe_analysis/AmBe_Data_Analysis/data/wcte_v11_20250513.json --output-dir /mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/nicf_data/data/analysis_files/final_pipeline_claude/1767

# Run RQE Calculation
python run_qe.py --thresh-min 2 --trms-cut 2.0 --source-pos-cm 0.0 152.5 0 --sig-parquet /mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/nicf_data/data/analysis_files/final_pipeline_claude/data/df_sig_R1767.parquet --bkg-parquet /mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/nicf_data/data/analysis_files/final_pipeline_claude/data/df_bkg_R1766.parquet --mc-npz /mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/1Mneutrons_NiCf_piFix_QGSP_BIC_HP_pos1767_CDSON_30901events_newTuning.npz --geo-file /mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/geofile_NuPRISMBeamTest_16cShort_mPMT.txt --output-dir /mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/nicf_data/data/analysis_files/final_pipeline_claude/1767
