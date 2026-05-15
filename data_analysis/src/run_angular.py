"""
NiCf Analysis Pipeline — Angular Acceptance
=============================================
Computes the angular acceptance per PMT by comparing hit rates
at different source positions, both in data and MC.

For each PMT:
    acceptance_data[pmt] = pure_signal_pos2[pmt] / pure_signal_ref[pmt]
    acceptance_mc[pmt]   = mc_hits_pos2[pmt] / mc_hits_ref[pmt]

Usage
-----
    python run_acceptance.py \
        --ref-sig ./output/pos1767/data/df_sig_R1767.parquet \
        --ref-mc  ./output/pos1767/data/relative_qe.parquet \
        --pos-sig ./output/pos1769/data/df_sig_R1769.parquet \
        --pos-mc  ./output/pos1769/data/relative_qe.parquet \
        --bkg     ./output/pos1767/data/df_bkg_R1766.parquet \
        --trms-cut 2.0 \
        --pos-label "R110 phi135 Z0" \
        --output-dir ./output/acceptance
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

N_PMTS = 2014


def parse_args():
    p = argparse.ArgumentParser(description="Angular acceptance calculation")
    p.add_argument("--ref-sig",   type=str, required=True, help="Reference position signal parquet")
    p.add_argument("--pos-sig",   type=str, required=True, help="New position signal parquet")
    p.add_argument("--bkg",       type=str, required=True, help="Background parquet (same for both)")
    p.add_argument("--ref-mc",    type=str, default=None,  help="Reference position relative_qe (with mc_hits column)")
    p.add_argument("--pos-mc",    type=str, default=None,  help="New position relative_qe (with mc_hits column)")
    p.add_argument("--trms-cut",  type=float, default=2.0)
    p.add_argument("--ref-label", type=str, default="Reference")
    p.add_argument("--pos-label", type=str, default="Position 2")
    p.add_argument("--output-dir", type=str, default="./output/acceptance")
    return p.parse_args()


def hits_per_pmt(pmt_ids):
    result = np.zeros(N_PMTS)
    ids, counts = np.unique(pmt_ids[pmt_ids < N_PMTS], return_counts=True)
    result[ids.astype(int)] = counts
    return result


def get_pure_signal(df_sig, df_bkg, trms_cut):
    """Apply tRMS cut, match events, subtract background, return hits per PMT and n_candidates."""
    df_s = df_sig[df_sig["trms"] < trms_cut]
    df_b = df_bkg[df_bkg["trms"] < trms_cut]

    common = np.intersect1d(df_s["event_id"].unique(), df_b["event_id"].unique())
    df_s = df_s[df_s["event_id"].isin(common)]
    df_b = df_b[df_b["event_id"].isin(common)]

    sig_hits = hits_per_pmt(df_s["pmt_id"].values)
    bkg_hits = hits_per_pmt(df_b["pmt_id"].values)
    pure = sig_hits - bkg_hits
    pure[pure < 0] = 0

    n_cands = df_s["candidate_id"].nunique() - df_b["candidate_id"].nunique()
    return pure, max(n_cands, 1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ─── 1. Load data ───
    print("Loading DataFrames...")
    df_bkg = pd.read_parquet(args.bkg)

    df_ref = pd.read_parquet(args.ref_sig)
    df_pos = pd.read_parquet(args.pos_sig)

    for df in [df_ref, df_pos, df_bkg]:
        if "trms" not in df.columns:
            df["trms"] = df.groupby("candidate_id")["hit_pmt_calibrated_times"].transform("std")

    # ─── 2. Compute pure signal for both positions ───
    print(f"\nComputing pure signal (tRMS < {args.trms_cut} ns)...")
    pure_ref, n_ref = get_pure_signal(df_ref, df_bkg, args.trms_cut)
    pure_pos, n_pos = get_pure_signal(df_pos, df_bkg, args.trms_cut)

    print(f"  {args.ref_label}: {n_ref} pure candidates")
    print(f"  {args.pos_label}: {n_pos} pure candidates")

    # ─── 3. Data acceptance ───
    rate_ref = pure_ref / n_ref
    rate_pos = pure_pos / n_pos

    acceptance_data = np.zeros(N_PMTS)
    err_acceptance_data = np.zeros(N_PMTS)
    mask = (rate_ref > 0) & (rate_pos > 0)
    acceptance_data[mask] = rate_pos[mask] / rate_ref[mask]

    # Error propagation: ratio of two Poisson rates
    err_pos = np.zeros(N_PMTS)
    err_ref = np.zeros(N_PMTS)
    err_pos[pure_pos > 0] = np.sqrt(pure_pos[pure_pos > 0]) / n_pos
    err_ref[pure_ref > 0] = np.sqrt(pure_ref[pure_ref > 0]) / n_ref
    err_acceptance_data[mask] = acceptance_data[mask] * np.sqrt(
        (err_pos[mask] / rate_pos[mask])**2 + (err_ref[mask] / rate_ref[mask])**2
    )

    # ─── 4. MC acceptance (if provided) ───
    acceptance_mc = None
    if args.ref_mc and args.pos_mc:
        print("\nComputing MC acceptance...")
        qe_ref = pd.read_parquet(args.ref_mc)
        qe_pos = pd.read_parquet(args.pos_mc)

        mc_ref = qe_ref["mc_hits"].values
        mc_pos = qe_pos["mc_hits"].values

        n_mc_ref = max(mc_ref.sum(), 1)
        n_mc_pos = max(mc_pos.sum(), 1)

        mc_rate_ref = mc_ref / n_mc_ref
        mc_rate_pos = mc_pos / n_mc_pos

        acceptance_mc = np.zeros(N_PMTS)
        mask_mc = (mc_rate_ref > 0) & (mc_rate_pos > 0)
        acceptance_mc[mask_mc] = mc_rate_pos[mask_mc] / mc_rate_ref[mask_mc]

    # ─── 5. Plots ───
    print("\nGenerating plots...")

    # Acceptance per PMT
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].errorbar(range(N_PMTS), acceptance_data, yerr=err_acceptance_data,
                     fmt="o", markersize=1.5, color="blue", ecolor="blue",
                     elinewidth=0.3, capsize=0, alpha=0.5,
                     label=f"Data: {args.pos_label} / {args.ref_label}")
    if acceptance_mc is not None:
        axes[0].scatter(range(N_PMTS), acceptance_mc, s=1.5, color="red", alpha=0.5,
                        label=f"MC: {args.pos_label} / {args.ref_label}")
    axes[0].axhline(1.0, linestyle=":", color="k")
    axes[0].set_ylabel("Acceptance (pos / ref)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Angular Acceptance: {args.pos_label} / {args.ref_label}")

    # Data/MC ratio of acceptances
    if acceptance_mc is not None:
        ratio = np.zeros(N_PMTS)
        mask_both = (acceptance_data > 0) & (acceptance_mc > 0)
        ratio[mask_both] = acceptance_data[mask_both] / acceptance_mc[mask_both]

        axes[1].scatter(range(N_PMTS), ratio, s=1.5, color="green", alpha=0.5,
                        label="Data acceptance / MC acceptance")
        axes[1].axhline(1.0, linestyle=":", color="k")
        axes[1].set_ylabel("Data / MC")
        axes[1].set_xlabel("PMT ID")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "angular_acceptance.png"), dpi=150)
    plt.close()

    # Projection histograms
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_data = acceptance_data[acceptance_data > 0]
    ax.hist(valid_data, bins=50, range=(0, 3), histtype="step",
            color="blue", linewidth=1.5,
            label=f"Data ($\\mu$={valid_data.mean():.2f}, $\\sigma$={valid_data.std():.2f})")
    if acceptance_mc is not None:
        valid_mc = acceptance_mc[acceptance_mc > 0]
        ax.hist(valid_mc, bins=50, range=(0, 3), histtype="step",
                color="red", linewidth=1.5,
                label=f"MC ($\\mu$={valid_mc.mean():.2f}, $\\sigma$={valid_mc.std():.2f})")
    ax.axvline(1.0, linestyle=":", color="k")
    ax.set_xlabel(f"Acceptance ({args.pos_label} / {args.ref_label})")
    ax.set_ylabel("PMTs")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "angular_acceptance_projection.png"), dpi=150)
    plt.close()

    # ─── 6. Save ───
    df_out = pd.DataFrame({
        "pmt_id": np.arange(N_PMTS),
        "acceptance_data": acceptance_data,
        "err_acceptance_data": err_acceptance_data,
        "rate_ref": rate_ref,
        "rate_pos": rate_pos,
    })
    if acceptance_mc is not None:
        df_out["acceptance_mc"] = acceptance_mc

    out_path = os.path.join(args.output_dir, "angular_acceptance.csv")
    df_out.to_csv(out_path, index=False)

    print(f"\n--- Acceptance Summary ---")
    print(f"  {args.ref_label}: {n_ref} pure candidates")
    print(f"  {args.pos_label}: {n_pos} pure candidates")
    valid = acceptance_data > 0
    print(f"  PMTs with valid acceptance: {valid.sum()} / {N_PMTS}")
    print(f"  Mean data acceptance: {acceptance_data[valid].mean():.3f} ± {acceptance_data[valid].std():.3f}")
    if acceptance_mc is not None:
        valid_mc = acceptance_mc > 0
        print(f"  Mean MC acceptance:   {acceptance_mc[valid_mc].mean():.3f} ± {acceptance_mc[valid_mc].std():.3f}")
    print(f"\nSaved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()