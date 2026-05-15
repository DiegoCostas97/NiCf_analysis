#!/usr/bin/env python3
"""
NiCf Analysis Pipeline — Vertex Reconstruction
=================================================
Reconstructs the interaction vertex for each candidate by fitting
the hit times to a point-source hypothesis using least squares.

For each candidate, minimises:
    χ² = Σᵢ (t_measured_i − t₀ − |vertex − PMT_i| / c_water)²

over 4 parameters: x, y, z (vertex position) and t₀ (emission time).

Usage
-----
    python run_reco.py \
        --sig-parquet ./output/data/df_sig_R1767.parquet \
        --geo-json /path/to/wcte_v11_20250513.json \
        --source-pos 0 1525 0 \
        --trms-cut 2.0 \
        --min-hits 4 \
        --output-dir ./output
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm

from functions import load_pmt_positions


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NiCf vertex reconstruction")
    p.add_argument("--sig-parquet", type=str, required=True,
                   help="Path to signal DataFrame parquet (no ToF correction)")
    p.add_argument("--geo-json",    type=str, required=True,
                   help="WCTE geometry JSON with PMT positions")
    p.add_argument("--source-pos",  type=float, nargs=3, default=[0, 1525, 0],
                   help="Source position [x, y, z] in mm (used as fit seed)")
    p.add_argument("--n-water",     type=float, default=1.33,
                   help="Water refractive index")
    p.add_argument("--trms-cut",    type=float, default=2.0,
                   help="Only reconstruct candidates with tRMS < this [ns]")
    p.add_argument("--min-hits",    type=int,   default=4,
                   help="Minimum hits per candidate to attempt reconstruction")
    p.add_argument("--output-dir",  type=str,   default="./output",
                   help="Output directory")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Reconstruction
# ═══════════════════════════════════════════════════════════════════

def build_pmt_position_array(pmt_positions, n_pmts=2014):
    """
    Build a (n_pmts, 3) array of PMT positions for fast lookup.
    """
    pos_array = np.zeros((n_pmts, 3))
    for pmt_id, pos in pmt_positions.items():
        if pmt_id < n_pmts:
            pos_array[pmt_id] = pos
    return pos_array


def residuals(params, hit_times, pmt_xyz, c_water_mm_ns):
    """
    Compute time residuals for the point-source hypothesis.

    params = [x, y, z, t0]
    residual_i = t_measured_i - t0 - |vertex - PMT_i| / c_water
    """
    vertex = params[:3]
    t0     = params[3]

    dxyz  = pmt_xyz - vertex
    dists = np.sqrt(np.sum(dxyz**2, axis=1))
    t_expected = t0 + dists / c_water_mm_ns

    return hit_times - t_expected


def reconstruct_candidate(hit_times, pmt_ids, pmt_pos_array,
                          c_water_mm_ns, seed_xyz, bounds=None):
    """
    Reconstruct vertex for a single candidate.

    Returns
    -------
    dict with vertex_x, vertex_y, vertex_z, vertex_t0, chi2, ndf, success
    """
    pmt_xyz = pmt_pos_array[pmt_ids]

    # Seed t0: earliest hit time minus ToF from seed to nearest PMT
    dists_seed = np.sqrt(np.sum((pmt_xyz - seed_xyz)**2, axis=1))
    t0_seed    = np.min(hit_times - dists_seed / c_water_mm_ns)

    x0 = np.array([seed_xyz[0], seed_xyz[1], seed_xyz[2], t0_seed])

    try:
        result = least_squares(
            residuals, x0,
            args=(hit_times, pmt_xyz, c_water_mm_ns),
            method='lm',
            max_nfev=1000,
        )

        chi2 = np.sum(result.fun**2)
        ndf  = len(hit_times) - 4  # 4 free parameters

        return {
            "vertex_x":  result.x[0],
            "vertex_y":  result.x[1],
            "vertex_z":  result.x[2],
            "vertex_t0": result.x[3],
            "chi2":      chi2,
            "ndf":       ndf,
            "chi2_ndf":  chi2 / ndf if ndf > 0 else np.nan,
            "success":   result.success,
        }

    except Exception:
        return {
            "vertex_x":  np.nan,
            "vertex_y":  np.nan,
            "vertex_z":  np.nan,
            "vertex_t0": np.nan,
            "chi2":      np.nan,
            "ndf":       np.nan,
            "chi2_ndf":  np.nan,
            "success":   False,
        }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    fig_dir  = os.path.join(args.output_dir, "figures")
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    c_water_mm_ns = (3e8 / args.n_water) * 1e-6  # mm/ns
    seed_xyz = np.array(args.source_pos)

    # ─── 1. Load PMT geometry ───
    print("Loading PMT geometry...")
    pmt_positions = load_pmt_positions(args.geo_json)
    pmt_pos_array = build_pmt_position_array(pmt_positions)
    print(f"  {len(pmt_positions)} PMTs loaded")

    # ─── 2. Load DataFrame ───
    print(f"\nLoading DataFrame from {args.sig_parquet}...")
    df = pd.read_csv(args.sig_parquet)

    # Compute tRMS if not present
    if "trms" not in df.columns:
        print("  Computing tRMS...")
        df["trms"] = df.groupby("candidate_id")[
            "hit_pmt_calibrated_times"].transform("std")

    if "nhits" not in df.columns:
        df["nhits"] = df.groupby("candidate_id")[
            "hit_pmt_calibrated_times"].transform("count")

    total_candidates = df["candidate_id"].nunique()
    print(f"  {len(df)} hits, {total_candidates} candidates")

    # ─── 3. Apply cuts ───
    print(f"\nApplying cuts: tRMS < {args.trms_cut} ns, nhits >= {args.min_hits}...")
    df_cut = df[(df["trms"] < args.trms_cut) & (df["nhits"] >= args.min_hits)]
    candidates_to_reco = df_cut["candidate_id"].unique()
    print(f"  {len(candidates_to_reco)} candidates to reconstruct "
          f"(of {total_candidates} total)")

    # ─── 4. Reconstruct ───
    print("\nReconstructing vertices...")
    results = []

    grouped = df_cut.groupby("candidate_id")
    for cand_id in tqdm(candidates_to_reco, total=len(candidates_to_reco)):
        group = grouped.get_group(cand_id)

        hit_times = group["hit_pmt_calibrated_times"].values.astype(np.float64)
        pmt_ids   = group["pmt_id"].values.astype(int)
        event_id  = group["event_id"].iloc[0]
        nhits     = len(hit_times)
        trms      = group["trms"].iloc[0]

        reco = reconstruct_candidate(
            hit_times, pmt_ids, pmt_pos_array,
            c_water_mm_ns, seed_xyz
        )

        reco["candidate_id"] = cand_id
        reco["event_id"]     = event_id
        reco["nhits"]        = nhits
        reco["trms"]         = trms
        results.append(reco)

    df_reco = pd.DataFrame(results)

    # ─── 5. Summary stats ───
    successful = df_reco[df_reco["success"] == True]
    failed     = df_reco[df_reco["success"] == False]
    print(f"\n--- Reconstruction Summary ---")
    print(f"  Total attempted:  {len(df_reco)}")
    print(f"  Successful:       {len(successful)} ({100*len(successful)/len(df_reco):.1f}%)")
    print(f"  Failed:           {len(failed)}")

    if len(successful) > 0:
        print(f"\n  Mean vertex (successful):")
        print(f"    X = {successful['vertex_x'].mean():.1f} ± {successful['vertex_x'].std():.1f} mm")
        print(f"    Y = {successful['vertex_y'].mean():.1f} ± {successful['vertex_y'].std():.1f} mm")
        print(f"    Z = {successful['vertex_z'].mean():.1f} ± {successful['vertex_z'].std():.1f} mm")
        print(f"  Mean χ²/ndf = {successful['chi2_ndf'].mean():.2f}")

    # ─── 6. Plots ───
    print("\nGenerating plots...")

    if len(successful) > 0:
        # Vertex positions
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(successful["vertex_x"], bins=100, color="black", edgecolor="white")
        axes[0].axvline(args.source_pos[0], color="red", linestyle="--", label="Source")
        axes[0].set_xlabel("Reco X [mm]"); axes[0].set_ylabel("Candidates")
        axes[0].set_title("Vertex X"); axes[0].legend()

        axes[1].hist(successful["vertex_y"], bins=100, color="black", edgecolor="white")
        axes[1].axvline(args.source_pos[1], color="red", linestyle="--", label="Source")
        axes[1].set_xlabel("Reco Y [mm]"); axes[1].set_ylabel("Candidates")
        axes[1].set_title("Vertex Y"); axes[1].legend()

        axes[2].hist(successful["vertex_z"], bins=100, color="black", edgecolor="white")
        axes[2].axvline(args.source_pos[2], color="red", linestyle="--", label="Source")
        axes[2].set_xlabel("Reco Z [mm]"); axes[2].set_ylabel("Candidates")
        axes[2].set_title("Vertex Z"); axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "reco_vertex_xyz.png"), dpi=150)
        plt.close()

        # Chi2/ndf
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(successful["chi2_ndf"], bins=100, range=(0, 10),
                color="black", edgecolor="white")
        ax.set_xlabel("$\\chi^2$ / ndf"); ax.set_ylabel("Candidates")
        ax.set_title("Reconstruction quality")
        ax.axvline(1.0, color="red", linestyle="--", label="$\\chi^2$/ndf = 1")
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "reco_chi2_ndf.png"), dpi=150)
        plt.close()

        # Distance from source
        dx = successful["vertex_x"] - args.source_pos[0]
        dy = successful["vertex_y"] - args.source_pos[1]
        dz = successful["vertex_z"] - args.source_pos[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(dist, bins=100, color="black", edgecolor="white")
        ax.set_xlabel("Distance reco vertex to source [mm]")
        ax.set_ylabel("Candidates")
        ax.set_title(f"Vertex resolution ($\\mu$={dist.mean():.1f} mm, "
                     f"$\\sigma$={dist.std():.1f} mm)")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "reco_vertex_resolution.png"), dpi=150)
        plt.close()

        # 2D scatter Y vs R
        r_reco = np.sqrt(successful["vertex_x"]**2 + successful["vertex_z"]**2)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(r_reco, successful["vertex_y"], s=1, alpha=0.1, color="black")
        ax.scatter([np.sqrt(args.source_pos[0]**2 + args.source_pos[2]**2)],
                   [args.source_pos[1]], color="red", s=100, marker="*",
                   zorder=10, label="Source position")
        ax.set_xlabel("Reco R [mm]"); ax.set_ylabel("Reco Y [mm]")
        ax.set_title("Reconstructed vertex positions")
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "reco_vertex_R_vs_Y.png"), dpi=150)
        plt.close()

    # ─── 7. Save ───
    reco_path = os.path.join(data_dir, "df_reco.csv")
    df_reco.to_csv(reco_path, index=False)
    print(f"\nSaved reconstructed vertices to {reco_path}")
    print("Done!")


if __name__ == "__main__":
    main()
