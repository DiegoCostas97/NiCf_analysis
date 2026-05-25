#!/usr/bin/env python3
"""
NiCf Analysis Pipeline — Run Analysis
=======================================
Reads signal and background runs, applies ToF correction,
runs the greedy nHits trigger, builds candidate DataFrames,
computes observables/purity and produces figures.

Usage
-----
    python run_analysis.py \
        --sig-run 1767 \
        --bkg-run 1766 \
        --n-parts 5 \
        --no-tof False
        --data-dir /path/to/raw_data/production_v0 \
        --geo-json /path/to/wcte_v11_20250513.json \
        --source-pos 0 1525 0 \
        --window 20 \
        --thresh-min 2 \
        --output-dir ./output
"""

import argparse
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from functions import (
    load_pmt_positions, build_tof_map, read_run,
    nHits_greedy, build_candidates_dataframe,
    add_candidate_observables, compute_purity,
    compute_charge_calibration, apply_charge_calibration,
)

# Append external tools path (adjust as needed)
sys.path.append(os.path.abspath(
    os.environ.get("WCTE_SOFTWARE_DIR",
                    "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk")
))

from WCTE_BRB_Data_Analysis.wcte.brbtools import sort_run_files, get_part_files

# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NiCf neutron capture analysis")
    p.add_argument("--sig-run",    type=int,   required=True,  help="Signal run number (e.g. 1767)")
    p.add_argument("--bkg-run",    type=int,   required=True,  help="Background run number (e.g. 1766)")
    p.add_argument("--n-parts",    type=int,   default=5,      help="Number of part files to process")
    p.add_argument("--data-dir",   type=str,   required=True,  help="Root data directory containing run folders")
    p.add_argument("--geo-json",   type=str,   required=True,  help="Path to WCTE geometry JSON")
    p.add_argument("--source-pos", type=float, nargs=3, default=[0, 1525, 0],
                   help="Source position [x, y, z] in mm")
    p.add_argument("--n-water",    type=float, default=1.33,   help="Water refractive index")
    p.add_argument("--window",     type=float, default=20,     help="Sliding window width [ns]")
    p.add_argument("--thresh-min", type=int,   default=2,      help="Minimum hits to trigger")
    p.add_argument("--output-dir", type=str,   default="./output", help="Output directory")
    p.add_argument("--mpmt-csv",   type=str,   default="/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCTE_event_display/mPMT_2D_projection_angles.csv",
                   help="Path to mPMT_2D_projection_angles.csv (for event displays)")
    p.add_argument("--wcsim-geo",  type=str,   default="/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/geofile_NuPRISMBeamTest_16cShort_mPMT.txt",
                   help="WCSim geofile with tube mapping (for event displays)")
    p.add_argument("--no-tof",    action="store_true", default=False,
                   help="Skip ToF correction (use raw calibrated times)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Plotting Helpers
# ═══════════════════════════════════════════════════════════════════

def plot_nhits_and_duration(sig_times, bkg_times, fig_dir):
    """N20 and candidate time length distributions."""
    nhits_sig, nhits_bkg = [], []
    dur_sig,   dur_bkg   = [], []

    for evC in sig_times.values():
        for c in evC:
            nhits_sig.append(len(c))
            dur_sig.append(c[-1] - c[0] if len(c) > 1 else 0)
    for evC in bkg_times.values():
        for c in evC:
            nhits_bkg.append(len(c))
            dur_bkg.append(c[-1] - c[0] if len(c) > 1 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(nhits_sig, bins=18, range=(2, 20), histtype="step",
             color="black", linewidth=1.5, label="SIG+BKG")
    ax1.hist(nhits_bkg, bins=18, range=(2, 20), histtype="step",
             color="blue",  linewidth=1.5, label="BKG")
    ax1.set_xlabel("N hits per candidate (N20)")
    ax1.set_ylabel("Candidates"); ax1.legend(); ax1.grid(True)

    ax2.hist(dur_sig, bins=100, range=(0, 20), histtype="step",
             color="black", linewidth=1.5, label="SIG+BKG")
    ax2.hist(dur_bkg, bins=100, range=(0, 20), histtype="step",
             color="blue",  linewidth=1.5, label="BKG")
    ax2.set_xlabel("Candidate time length [ns]")
    ax2.set_ylabel("Candidates"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "nhits_and_duration.png"), dpi=150)
    plt.close()


def plot_trms(sig_times, bkg_times, sig_times_raw, bkg_times_raw, fig_dir):
    """tRMS distributions with and without ToF correction."""
    trms_sig, trms_bkg = [], []
    for evC in sig_times.values():
        for c in evC: trms_sig.append(np.std(c))
    for evC in bkg_times.values():
        for c in evC: trms_bkg.append(np.std(c))

    trms_sig_raw, trms_bkg_raw = [], []
    for evC in sig_times_raw.values():
        for c in evC: trms_sig_raw.append(np.std(c))
    for evC in bkg_times_raw.values():
        for c in evC: trms_bkg_raw.append(np.std(c))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(trms_sig, bins=500, range=(0, 10), histtype="step",
            color="black", linewidth=1.5,
            label=f"SIG+BKG ToF-corr ($\\mu$={np.mean(trms_sig):.2f} ns)")
    ax.hist(trms_bkg, bins=500, range=(0, 10), histtype="step",
            color="blue", linewidth=1.5,
            label=f"BKG ToF-corr ($\\mu$={np.mean(trms_bkg):.2f} ns)")
    ax.hist(trms_sig_raw, bins=500, range=(0, 10), histtype="step",
            color="gray", linewidth=1.5, linestyle="--",
            label=f"SIG+BKG raw ($\\mu$={np.mean(trms_sig_raw):.2f} ns)")
    ax.hist(trms_bkg_raw, bins=500, range=(0, 10), histtype="step",
            color="lightblue", linewidth=1.5, linestyle="--",
            label=f"BKG raw ($\\mu$={np.mean(trms_bkg_raw):.2f} ns)")
    ax.set_xlabel("$t_{RMS}$ [ns]"); ax.set_ylabel("Candidates")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "trms.png"), dpi=150)
    plt.close()

    # ─── Second plot: fit gaussian to SIG+BKG peak ───
    from scipy.optimize import curve_fit
    from scipy.stats import norm

    def gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    counts, edges = np.histogram(trms_sig, bins=500, range=(0, 10))
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # Fit around the peak
    peak_idx = np.argmax(counts)
    peak_x = centers[peak_idx]
    fit_mask = (centers > max(0, peak_x - 0.6)) & (centers < peak_x + 0.6)

    try:
        popt, _ = curve_fit(gaussian, centers[fit_mask], counts[fit_mask],
                            p0=[counts[peak_idx], peak_x, 0.3])
        mu_fit, sigma_fit = popt[1], abs(popt[2])
    except Exception:
        mu_fit, sigma_fit = peak_x, np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(trms_sig, bins=500, range=(0, 10), histtype="step",
            color="black", linewidth=1.5,
            label=f"SIG+BKG (peak $\\mu$={mu_fit:.2f} ns, $\\sigma$={sigma_fit:.2f})")
    ax.hist(trms_bkg, bins=500, range=(0, 10), histtype="step",
            color="blue", linewidth=1.5,
            label=f"BKG ($\\mu$={np.mean(trms_bkg):.2f} ns)")

    # Draw fit
    x_fit = np.linspace(max(0, mu_fit - 3*sigma_fit), mu_fit + 3*sigma_fit, 200)
    ax.plot(x_fit, gaussian(x_fit, *popt), color="red", linewidth=1.5,
            label="Gaussian fit to SIG+BKG peak")

    ax.set_xlabel("$t_{RMS}$ [ns]"); ax.set_ylabel("Candidates")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "trms_fitted.png"), dpi=150)
    plt.close()


def plot_tc(sig_times, bkg_times, fig_dir):
    """Mean hit time (tc) distribution."""
    tc_sig, tc_bkg = [], []
    for evC in sig_times.values():
        for c in evC:
            t = np.array(c)
            tc_sig.append((t - t.min()).mean())
    for evC in bkg_times.values():
        for c in evC:
            t = np.array(c)
            tc_bkg.append((t - t.min()).mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(tc_sig, bins=500, range=(0, 20), histtype="step",
            color="black", linewidth=1.5, label="SIG+BKG")
    ax.hist(tc_bkg, bins=500, range=(0, 20), histtype="step",
            color="blue", linewidth=1.5, label="BKG")
    ax.set_xlabel("$t_c$ [ns]"); ax.set_ylabel("Candidates")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "tc.png"), dpi=150)
    plt.close()


def plot_nhits_vs_trms_2d(sig_times, bkg_times, fig_dir):
    """2D histogram of N20 vs tRMS."""
    def collect(times_dict):
        nhits, trms = [], []
        for evC in times_dict.values():
            for c in evC:
                nhits.append(len(c))
                trms.append(np.std(c))
        return np.array(nhits), np.array(trms)

    nhits_s, trms_s = collect(sig_times)
    nhits_b, trms_b = collect(bkg_times)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    from matplotlib.colors import LogNorm

    ax1.hist2d(trms_s, nhits_s, bins=[200, 50], range=[[0, 10], [2, 50]],
               norm=LogNorm()); ax1.set_title("SIG+BKG")
    ax1.set_xlabel("$t_{RMS}$ [ns]"); ax1.set_ylabel("N20 [hits]")

    ax2.hist2d(trms_b, nhits_b, bins=[200, 50], range=[[0, 10], [2, 50]],
               norm=LogNorm()); ax2.set_title("BKG")
    ax2.set_xlabel("$t_{RMS}$ [ns]"); ax2.set_ylabel("N20 [hits]")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "nhits_vs_trms_2d.png"), dpi=150)
    plt.close()


def plot_purity_histograms(df_sig, df_bkg, fig_dir):
    """Candidates per event after all cuts."""
    cands_sig = df_sig.groupby("event_id")["candidate_id"].nunique().values
    cands_bkg = df_bkg.groupby("event_id")["candidate_id"].nunique().values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cands_sig, bins=50, histtype="step", color="black",
            linewidth=1.5, label="SIG+BKG")
    ax.hist(cands_bkg, bins=50, histtype="step", color="blue",
            linewidth=1.5, label="BKG")
    ax.set_xlabel("Candidates per event")
    ax.set_ylabel("Events"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "candidates_per_event.png"), dpi=150)
    plt.close()


def plot_charge_per_pmt(df_sig, fig_dir, suffix=""):
    """Mean charge per PMT across all candidates."""
    mean_charge = df_sig.groupby("pmt_id")["hit_pmt_charges"].mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(mean_charge.index, mean_charge.values,
                yerr=df_sig.groupby("pmt_id")["hit_pmt_charges"].std().values,
                fmt='.', markersize=2, color='black', ecolor='black',
                elinewidth=0.3, capsize=0, label="Charge per PMT [ADCs]")
    ax.axhline(mean_charge.mean(), color='red', linestyle='--',
               label=f"Mean = {mean_charge.mean():.1f} ADC")
    ax.set_xlabel("PMT ID"); ax.set_ylabel("Charge [ADCs]")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"charge_per_pmt{suffix}.png"), dpi=150)
    plt.close()


def plot_charge_per_pmt_calibrated(df_sig, fig_dir):
    """Mean calibrated charge per PMT — should be flat after calibration."""
    if "hit_pmt_charges_calibrated" not in df_sig.columns:
        return

    mean_charge = df_sig.groupby("pmt_id")["hit_pmt_charges_calibrated"].mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(mean_charge.index, mean_charge.values,
                yerr=df_sig.groupby("pmt_id")["hit_pmt_charges_calibrated"].std().values,
                fmt='.', markersize=2, color='black', ecolor='black',
                elinewidth=0.3, capsize=0, label="Calibrated charge per PMT [ADCs]")
    ax.axhline(mean_charge.mean(), color='red', linestyle='--',
               label=f"Mean = {mean_charge.mean():.1f} ADC")
    ax.set_xlabel("PMT ID"); ax.set_ylabel("Calibrated Charge [ADCs]")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "charge_per_pmt_calibrated.png"), dpi=150)
    plt.close()


def plot_event_displays(df_sig, df_bkg, fig_dir,
                        mpmt_positions_csv, wcsim_geofile):
    """
    2D event display of hits per PMT for SIG and BKG runs.
    """
    try:
        import matplotlib.colors as colors
        from WCTE_event_display.EventDisplay import EventDisplay
    except ImportError:
        print("  WCTE_event_display not available, skipping event displays")
        return

    n_pmts = 2014

    eventDisplay = EventDisplay()
    eventDisplay.load_mPMT_positions(mpmt_positions_csv)
    eventDisplay.load_wcsim_tubeno_mapping(wcsim_geofile)

    # --- Compute hits per PMT for SIG (sets scale) ---
    nhits_sig = (df_sig.groupby("pmt_id")["hit_pmt_charges"].count()
                 .reindex(range(n_pmts), fill_value=0).values.astype(float))
    vmin = 1e-1
    vmax = nhits_sig.max()

    # --- SIG display ---
    eventDisplay.plotEventDisplay(
        nhits_sig,
        color_norm=colors.Normalize(vmin=vmin, vmax=vmax),
        style="dark_background"
    )
    plt.title("SIG+BKG run — hits per PMT")
    plt.savefig(os.path.join(fig_dir, "event_display_sig.png"),
                dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()

    # --- BKG display (same scale) ---
    nhits_bkg = (df_bkg.groupby("pmt_id")["hit_pmt_charges"].count()
                 .reindex(range(n_pmts), fill_value=0).values.astype(float))
    eventDisplay.plotEventDisplay(
        nhits_bkg,
        color_norm=colors.Normalize(vmin=vmin, vmax=vmax),
        style="dark_background"
    )
    plt.title("BKG run — hits per PMT")
    plt.savefig(os.path.join(fig_dir, "event_display_bkg.png"),
                dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()

    plt.style.use('default')


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Directories
    fig_dir = os.path.join(args.output_dir, "figures")
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ─── 1. Build ToF map ───
    print("Loading PMT geometry and building ToF map...")
    if args.no_tof:
        print("No ToF Correction will be applied. You enabled --no-tof argument")
    pmt_positions = load_pmt_positions(args.geo_json)
    tof_map = build_tof_map(pmt_positions, args.source_pos, args.n_water)
    print(f"  {len(pmt_positions)} PMTs, source at {args.source_pos}")

    # ─── 2. Read signal run ───
    print(f"\nReading SIG+BKG run {args.sig_run}...")
    sig_files = sort_run_files(
        f"{args.data_dir}/{args.sig_run}/WCTE_offline_R{args.sig_run}S*P*.root"
    )
    sig_parts = get_part_files(sig_files)
    sig_data = read_run(args.sig_run, sig_files, sig_parts,
                        args.n_parts, tof_map=None if args.no_tof else tof_map)
    print(f"  {len(sig_data['hit_times'])} events read")

    # ─── 3. Read background run ───
    print(f"\nReading BKG run {args.bkg_run}...")
    bkg_files = sort_run_files(
        f"{args.data_dir}/{args.bkg_run}/WCTE_offline_R{args.bkg_run}S*P*.root"
    )
    bkg_parts = get_part_files(bkg_files)
    bkg_data = read_run(args.bkg_run, bkg_files, bkg_parts,
                        args.n_parts, tof_map=None if args.no_tof else tof_map)
    print(f"  {len(bkg_data['hit_times'])} events read")

    # ─── 3b. Read RAW data (without ToF) for tRMS comparison ───
    print(f"\nReading RAW data (no ToF) for tRMS comparison plot...")
    sig_data_raw = read_run(args.sig_run, sig_files, sig_parts,
                            args.n_parts, tof_map=None)
    bkg_data_raw = read_run(args.bkg_run, bkg_files, bkg_parts,
                            args.n_parts, tof_map=None)

    # ─── 4. Run trigger ───
    print(f"\nRunning greedy nHits trigger (w={args.window} ns, "
          f"thresh_min={args.thresh_min})...")

    print("  SIG+BKG...")
    sig_indices, sig_times, sig_seeds = nHits_greedy(
        sig_data['hit_times'], args.window, args.thresh_min
    )
    print("  BKG...")
    bkg_indices, bkg_times, bkg_seeds = nHits_greedy(
        bkg_data['hit_times'], args.window, args.thresh_min
    )

    # Trigger on raw times (for tRMS comparison)
    print("  SIG+BKG raw...")
    _, sig_times_raw, _ = nHits_greedy(
        sig_data_raw['hit_times'], args.window, args.thresh_min
    )
    print("  BKG raw...")
    _, bkg_times_raw, _ = nHits_greedy(
        bkg_data_raw['hit_times'], args.window, args.thresh_min
    )

    # ─── 5. Plots (no cuts) ───
    print("\nGenerating plots (no cuts)...")
    plot_nhits_and_duration(sig_times, bkg_times, fig_dir)
    plot_trms(sig_times, bkg_times, sig_times_raw, bkg_times_raw, fig_dir)    
    plot_tc(sig_times, bkg_times, fig_dir)
    plot_nhits_vs_trms_2d(sig_times, bkg_times, fig_dir)

    # ─── 6. Purity (no cuts) ───
    n_ev_sig = len(sig_data['hit_times'])
    n_ev_bkg = len(bkg_data['hit_times'])

    purity_no_cuts = compute_purity(sig_times, bkg_times, n_ev_sig, n_ev_bkg)
    print(f"\n--- Purity (no cuts) ---")
    print(f"  Mean candidates/event SIG+BKG: {purity_no_cuts['mean_sig']:.4f}")
    print(f"  Mean candidates/event BKG:     {purity_no_cuts['mean_bkg']:.4f}")
    print(f"  Purity: {purity_no_cuts['purity']:.2f}%")

    # ─── 7. Build DataFrames (no cuts on trigger) ───
    print("\nBuilding candidate DataFrames...")
    df_sig = build_candidates_dataframe(sig_indices, sig_times, sig_data, "sig")
    df_bkg = build_candidates_dataframe(bkg_indices, bkg_times, bkg_data, "bkg")

    print("  Adding per-candidate observables...")
    df_sig = add_candidate_observables(df_sig)
    df_bkg = add_candidate_observables(df_bkg)

    # ─── 7b. Add raw (non-ToF-corrected) times ───
    if not args.no_tof:
        print("  Adding raw (non-ToF-corrected) hit times...")
        df_sig["hit_pmt_calibrated_times_raw"] = (
            df_sig["hit_pmt_calibrated_times"] + df_sig["pmt_id"].map(tof_map)
        )
        df_sig["trms_raw"] = df_sig.groupby("candidate_id")[
            "hit_pmt_calibrated_times_raw"].transform("std")

        df_bkg["hit_pmt_calibrated_times_raw"] = (
            df_bkg["hit_pmt_calibrated_times"] + df_bkg["pmt_id"].map(tof_map)
        )
        df_bkg["trms_raw"] = df_bkg.groupby("candidate_id")[
            "hit_pmt_calibrated_times_raw"].transform("std")
    else:
        # Times are already raw, duplicate for consistency
        df_sig["hit_pmt_calibrated_times_raw"] = df_sig["hit_pmt_calibrated_times"]
        df_sig["trms_raw"] = df_sig["trms"]
        df_bkg["hit_pmt_calibrated_times_raw"] = df_bkg["hit_pmt_calibrated_times"]
        df_bkg["trms_raw"] = df_bkg["trms"]

    # ─── 8. Charge Calibration ───
    print("\nComputing charge calibration...")
    cal_factors, global_mean = compute_charge_calibration(df_sig)
    print(f"  Global mean charge: {global_mean:.2f} ADC")

    df_sig = apply_charge_calibration(df_sig, cal_factors)
    df_bkg = apply_charge_calibration(df_bkg, cal_factors)

    # ─── 9. Plots ───
    plot_purity_histograms(df_sig, df_bkg, fig_dir)
    plot_charge_per_pmt(df_sig, fig_dir, suffix="_raw")
    plot_charge_per_pmt_calibrated(df_sig, fig_dir)

    if args.mpmt_csv is not None and args.wcsim_geo is not None:
        print("  Generating event displays...")
        plot_event_displays(df_sig[df_sig["trms"].values <=2], df_bkg[df_bkg["trms"].values <=2], fig_dir,
                            args.mpmt_csv, args.wcsim_geo)

    # ─── 10. Save DataFrames ───
    sig_path = os.path.join(data_dir, f"df_sig_R{args.sig_run}.parquet")
    bkg_path = os.path.join(data_dir, f"df_bkg_R{args.bkg_run}.parquet")
    df_sig.to_parquet(sig_path, index=False)
    df_bkg.to_parquet(bkg_path, index=False)
    print(f"\nSaved DataFrames:")
    print(f"  {sig_path} ({len(df_sig)} hits)")
    print(f"  {bkg_path} ({len(df_bkg)} hits)")

    # ─── 10. Summary ───
    summary = {
        "sig_run": args.sig_run,
        "bkg_run": args.bkg_run,
        "n_parts": args.n_parts,
        "window_ns": args.window,
        "thresh_min": args.thresh_min,
        "source_pos_mm": args.source_pos,
        "n_water": args.n_water,
        "n_events_sig": n_ev_sig,
        "n_events_bkg": n_ev_bkg,
        "purity_no_cuts": purity_no_cuts,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nDone!")


if __name__ == "__main__":
    main()
