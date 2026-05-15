# NiCf Analysis Pipeline

Modularised analysis pipeline for neutron capture signal identification
in the WCTE Water Cherenkov detector using a NiCf source.

## Structure

```
src/
├── functions.py       # All reusable functions (I/O, trigger, ToF, purity)
├── run_analysis.py    # Main pipeline: read → trigger → DataFrames → plots
├── run_qe.py          # QE studies: data vs MC per-PMT comparison
└── README.md
```

## Quick Start

### 1. Run the main analysis

```bash
python src/run_analysis.py \
    --sig-run 1767 \
    --bkg-run 1766 \
    --n-parts 5 \
    --data-dir /path/to/raw_data/production_v0 \
    --geo-json /path/to/wcte_v11_20250513.json \
    --source-pos 0 1525 0 \
    --window 20 \
    --thresh-min 2 \
    --output-dir ./output
```

This will:
- Read signal and background runs from ROOT files
- Apply ToF correction to the source position
- Run the greedy nHits trigger
- Build hit-level DataFrames (saved as parquet)
- Generate diagnostic plots in `output/figures/`
- Compute purity and save a summary JSON

### 2. Run QE studies

```bash
python src/run_qe.py \
    --sig-parquet ./output/data/df_sig_R1767.parquet \
    --bkg-parquet ./output/data/df_bkg_R1766.parquet \
    --mc-file /path/to/wcsim_nicf.npz \
    --geo-file /path/to/geofile_NuPRISMBeamTest_16cShort_mPMT.txt \
    --trms-cut 2.0 \
    --output-dir ./output
```

## Arguments

### run_analysis.py

| Argument       | Default       | Description                          |
|----------------|---------------|--------------------------------------|
| `--sig-run`    | (required)    | Signal run number (e.g. 1767)        |
| `--bkg-run`    | (required)    | Background run number (e.g. 1766)    |
| `--n-parts`    | 5             | Number of part files to process      |
| `--data-dir`   | (required)    | Root data directory with run folders  |
| `--geo-json`   | (required)    | WCTE geometry JSON                   |
| `--source-pos` | 0 1525 0      | Source position [x, y, z] in mm      |
| `--n-water`    | 1.33          | Water refractive index               |
| `--window`     | 20            | Sliding window width [ns]            |
| `--thresh-min` | 2             | Minimum hits to trigger              |
| `--output-dir` | ./output      | Output directory                     |

### run_qe.py

| Argument         | Default  | Description                        |
|------------------|----------|------------------------------------|
| `--sig-parquet`  | (required) | Signal DataFrame parquet         |
| `--bkg-parquet`  | (required) | Background DataFrame parquet     |
| `--mc-file`      | (required) | WCSim MC .npz file               |
| `--geo-file`     | (required) | WCSim geofile for tube mapping   |
| `--trms-cut`     | 2.0      | tRMS cut [ns] for QE selection     |
| `--output-dir`   | ./output | Output directory                   |

## Output

```
output/
├── summary.json              # Run parameters and purity results
├── data/
│   ├── df_sig_R1767.parquet  # Signal candidate DataFrame
│   ├── df_bkg_R1766.parquet  # Background candidate DataFrame
│   └── relative_qe.csv      # Per-PMT relative QE values
└── figures/
    ├── nhits_and_duration.png
    ├── trms.png
    ├── tc.png
    ├── nhits_vs_trms_2d.png
    ├── candidates_per_event.png
    ├── charge_per_pmt.png
    └── relative_qe.png
```

## Dependencies

- numpy, pandas, matplotlib, awkward, uproot, tqdm
- WCTE_BRB_Data_Analysis (for sort_run_files, get_part_files)
- scipy (optional, for fits)

## Environment Variable

Set `WCTE_SOFTWARE_DIR` to point to the WCTE software installation:
```bash
export WCTE_SOFTWARE_DIR=/path/to/hk/software
```
