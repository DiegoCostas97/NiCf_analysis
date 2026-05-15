#!/bin/bash

BASE="/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/nicf_data/data/analysis_files/final_pipeline_claude"
REF="1769"
BKG="${BASE}/1767/data/df_bkg_R1766.parquet"
REF_SIG="${BASE}/${REF}/data/df_sig_R${REF}.parquet"
REF_MC="${BASE}/${REF}/data/relative_qe.parquet"
REF_LABEL="Run 1769 (0.0, 51.2, 1175.8)"

# Run 1767 vs 1769
python run_angular.py \
    --ref-sig ${REF_SIG} \
    --pos-sig ${BASE}/1767/data/df_sig_R1767.parquet \
    --bkg ${BKG} \
    --trms-cut 2.0 \
    --ref-label "${REF_LABEL}" \
    --pos-label "Run 1767 (0, 1525, 0)" \
    --output-dir ${BASE}/angular_${REF}_vs_1767 \
    --ref-mc ${REF_MC} \
    --pos-mc ${BASE}/1767/data/relative_qe.parquet

# Run 2336 vs 1769
python run_angular.py \
    --ref-sig ${REF_SIG} \
    --pos-sig ${BASE}/2336/data/df_sig_R2336.parquet \
    --bkg ${BKG} \
    --trms-cut 2.0 \
    --ref-label "${REF_LABEL}" \
    --pos-label "Run 2336 (0, -441, 563)" \
    --output-dir ${BASE}/angular_${REF}_vs_2336 \
    --ref-mc ${REF_MC} \
    --pos-mc ${BASE}/2336/data/relative_qe.parquet