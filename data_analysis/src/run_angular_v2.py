import json
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 1. Load geometry ───
with open("/home/usc/ie/dcr/hk/ambe_analysis/AmBe_Data_Analysis/data/wcte_v11_20250513.json") as f:
    geo_data = json.load(f)

mpmt_data = geo_data["mpmts"]
N_PMTS = 2014

pmt_pos    = np.zeros((N_PMTS, 3))
pmt_normal = np.zeros((N_PMTS, 3))

for mpmt_idx in sorted(mpmt_data.keys(), key=int):
    mpmt = mpmt_data[mpmt_idx]
    for pmt_idx in sorted(mpmt["pmts"].keys(), key=int):
        pmt = mpmt["pmts"][pmt_idx]
        pmt_id = int(mpmt_idx) * 19 + int(pmt_idx)
        if pmt_id < N_PMTS:
            pmt_pos[pmt_id]    = pmt["placement"]["location"]
            pmt_normal[pmt_id] = pmt["placement"]["direction_z"]

# ─── 2. For each source position, compute cos(theta) and R per PMT ───
source_positions = {
    "R1767": np.array([0, 1525, 0]),
    "R1769": np.array([0, 476, 1175.8]),
    "R2336": np.array([0, -16.64, 562.8]),
    "R2337": np.array([-476.4, -28.1375, 291.7])
}

def compute_angles(source_pos):
    """For each PMT, compute cos(theta) and distance R to source."""
    # Vector from source to PMT
    vec = pmt_pos - source_pos  # (N_PMTS, 3)
    R = np.linalg.norm(vec, axis=1)  # distance
    
    # Normalize
    vec_norm = vec / R[:, None]
    
    # cos(theta) = dot(source_to_pmt_normalized, pmt_normal)
    # Note: normal points outward, light comes inward, so we use -vec
    cos_theta = np.sum(-vec_norm * pmt_normal, axis=1)
    
    return cos_theta, R

# ─── 3. Load pure signal hits per PMT (from run_qe output or compute here) ───
def get_pure_hits(df_sig_path, df_bkg_path, trms_cut=2.0):
    df_sig = pd.read_parquet(df_sig_path)
    df_bkg = pd.read_parquet(df_bkg_path)
    
    if "trms" not in df_sig.columns:
        df_sig["trms"] = df_sig.groupby("candidate_id")["hit_pmt_calibrated_times"].transform("std")
    if "trms" not in df_bkg.columns:
        df_bkg["trms"] = df_bkg.groupby("candidate_id")["hit_pmt_calibrated_times"].transform("std")
    
    df_s = df_sig[df_sig["trms"] < trms_cut]
    df_b = df_bkg[df_bkg["trms"] < trms_cut]
    
    common = np.intersect1d(df_s["event_id"].unique(), df_b["event_id"].unique())
    df_s = df_s[df_s["event_id"].isin(common)]
    df_b = df_b[df_b["event_id"].isin(common)]
    
    hits_s = np.zeros(N_PMTS)
    hits_b = np.zeros(N_PMTS)
    
    for pmt_id, cnt in df_s.groupby("pmt_id").size().items():
        if 0 <= pmt_id < N_PMTS:
            hits_s[int(pmt_id)] = cnt
    for pmt_id, cnt in df_b.groupby("pmt_id").size().items():
        if 0 <= pmt_id < N_PMTS:
            hits_b[int(pmt_id)] = cnt
    
    pure = hits_s - hits_b
    pure[pure < 0] = 0
    n_cands = df_s["candidate_id"].nunique() - df_b["candidate_id"].nunique()
    return pure, max(n_cands, 1)

# ─── Example for Run 1767 ───
BASE = "/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/nicf_data/data/analysis_files/final_pipeline_claude"
BKG = f"{BASE}/1767/data/df_bkg_R1766.parquet"

runs = {
    "R1767": f"{BASE}/1767/data/df_sig_R1767.parquet",
    "R1769": f"{BASE}/1769/data/df_sig_R1769.parquet",
    "R2336": f"{BASE}/2336/data/df_sig_R2336.parquet",
    "R2337": f"{BASE}/2337/data/df_sig_R2337.parquet"
}

# ─── 4. Load mPMT type info (detailed) ───
with open("/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/raw_data/other_mpmt_info.dict", "rb") as f:
    mpmt_info = pickle.load(f)

pmt_category = np.full(N_PMTS, "Other", dtype=object)

for pmt_id in range(N_PMTS):
    slot = pmt_id // 19
    if slot in mpmt_info:
        mtype = mpmt_info[slot].get("mpmt_type", "")
        msite = mpmt_info[slot].get("mpmt_site", "")
        if msite == "TRI" and mtype == "In-situ":
            pmt_category[pmt_id] = "TRI In-situ"
        elif msite == "TRI" and mtype == "Ex-situ":
            pmt_category[pmt_id] = "TRI Ex-situ"
        elif msite == "WUT" and mtype == "In-situ":
            pmt_category[pmt_id] = "WUT In-situ"
        elif msite == "WUT" and mtype == "Ex-situ":
            pmt_category[pmt_id] = "WUT Ex-situ"

# Keep is_insitu for backward compatibility
is_insitu = np.array([c in ["TRI In-situ", "WUT In-situ"] for c in pmt_category])

category_colors = {
    "TRI In-situ": "blue",
    "TRI Ex-situ": "green",
    "WUT In-situ": "red",
    "WUT Ex-situ": "orange",
}

# ─── 5. Plot A(theta) = N_i * R_i^2 vs cos(theta) for each run ───
fig, axes = plt.subplots(1, len(runs), figsize=(6*len(runs), 6))
if len(runs) == 1:
    axes = [axes]

for ax, (run_label, sig_path) in zip(axes, runs.items()):
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    
    # Normalize hits per candidate
    N_i = pure_hits / n_cands
    
    # A(theta) proxy = N_i * R_i^2
    A_theta = N_i * R**2
    
    valid = (N_i > 0) & (cos_theta > 0)  # only PMTs facing the source
    
    ax.scatter(cos_theta[valid & is_insitu], A_theta[valid & is_insitu],
               s=3, alpha=0.5, color="red", label="In-situ")
    ax.scatter(cos_theta[valid & ~is_insitu], A_theta[valid & ~is_insitu],
               s=3, alpha=0.5, color="blue", label="Ex-situ")
    ax.set_xlabel("cos θ")
    ax.set_ylabel("$N_i \\times R_i^2$")
    ax.set_title(f"{run_label}\nsource = {source.tolist()}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_all_runs.png", dpi=150)
plt.close()

# ─── 6. Combined: all runs on same plot ───
fig, ax = plt.subplots(figsize=(10, 7))
colors_run = {"R1767": "black", "R1769": "blue", "R2336": "green", "R2337": "red"}

for run_label, sig_path in runs.items():
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    
    N_i = pure_hits / n_cands
    A_theta = N_i * R**2
    valid = (N_i > 0) & (cos_theta > 0)
    
    ax.scatter(cos_theta[valid], A_theta[valid],
               s=3, alpha=0.4, color=colors_run[run_label], label=run_label)

ax.set_xlabel("cos θ")
ax.set_ylabel("$N_i \\times R_i^2$")
ax.set_title("PMT Angular Response — All source positions")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_combined.png", dpi=150)
plt.close()

# ─── 7. 2D histogram like Ka Ming's (combined) ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for run_label, sig_path in runs.items():
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    N_i = pure_hits / n_cands
    A_theta = N_i * R**2
    valid = (N_i > 0) & (cos_theta > 0)
    
    mask_ex = valid & ~is_insitu
    mask_in = valid & is_insitu
    
    if run_label == list(runs.keys())[0]:  # first run sets the hist2d
        all_cos_ex, all_A_ex = cos_theta[mask_ex], A_theta[mask_ex]
        all_cos_in, all_A_in = cos_theta[mask_in], A_theta[mask_in]
    else:
        all_cos_ex = np.concatenate([all_cos_ex, cos_theta[mask_ex]])
        all_A_ex   = np.concatenate([all_A_ex, A_theta[mask_ex]])
        all_cos_in = np.concatenate([all_cos_in, cos_theta[mask_in]])
        all_A_in   = np.concatenate([all_A_in, A_theta[mask_in]])

from matplotlib.colors import LogNorm
ax1.hist2d(all_cos_ex, all_A_ex, bins=[20, 20], norm=LogNorm())
ax1.set_xlabel("cos θ"); ax1.set_ylabel("$N_i R_i^2$"); ax1.set_title("Ex-situ")

ax2.hist2d(all_cos_in, all_A_in, bins=[20, 20], norm=LogNorm())
ax2.set_xlabel("cos θ"); ax2.set_ylabel("$N_i R_i^2$"); ax2.set_title("In-situ")

plt.suptitle("PMT Angular Response — All positions combined")
plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_2d.png", dpi=150)
plt.close()

# ─── 7b. 2D histogram per run (all mPMTs together) ───
fig, axes = plt.subplots(1, len(runs), figsize=(6*len(runs), 6))
if len(runs) == 1:
    axes = [axes]

from matplotlib.colors import LogNorm

for ax, (run_label, sig_path) in zip(axes, runs.items()):
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    N_i = pure_hits / n_cands
    A_theta = N_i * R**2
    valid = (N_i > 0) & (cos_theta > 0)
    
    ax.hist2d(cos_theta[valid], A_theta[valid], bins=[20, 20], norm=LogNorm())
    ax.set_xlabel("cos θ"); ax.set_ylabel("$N_i R_i^2$")
    ax.set_title(f"{run_label}\nsource = {source_positions[run_label].tolist()}")

plt.suptitle("PMT Angular Response 2D — per run (all mPMTs)")
plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_2d_per_run.png", dpi=150)
plt.close()

# ─── 8. Angular response: Data vs MC ───

n_cands_data = {"R1767": 516193, "R1769": 77699, "R2336": 100877, "R2337": 193175}
n_cands_mc   = {"R1767": 45818,  "R1769": 49548, "R2336": 43446,  "R2337": 43316}

mc_files = {
    "R1767": f"{BASE}/1767/data/relative_qe.parquet",
    "R1769": f"{BASE}/1769/data/relative_qe.parquet",
    "R2336": f"{BASE}/2336/data/relative_qe.parquet",
    "R2337": f"{BASE}/2337/data/relative_qe.parquet"
}

# Per-run: Data vs MC
fig, axes = plt.subplots(1, len(runs), figsize=(6*len(runs), 6))
if len(runs) == 1:
    axes = [axes]

for ax, (run_label, sig_path) in zip(axes, runs.items()):
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    
    # Data
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    N_i_data = pure_hits / n_cands
    A_data = N_i_data * R**2
    
    # MC
    qe_df = pd.read_parquet(mc_files[run_label])
    mc_hits = qe_df["mc_hits"].values
    N_i_mc = mc_hits / n_cands_mc[run_label]
    A_mc = N_i_mc * R**2
    
    valid_d = (N_i_data > 0) & (cos_theta > 0)
    valid_m = (N_i_mc > 0) & (cos_theta > 0)
    
    ax.scatter(cos_theta[valid_d], A_data[valid_d],
               s=3, alpha=0.4, color="blue", label="Data")
    ax.scatter(cos_theta[valid_m], A_mc[valid_m],
               s=3, alpha=0.4, color="red", label="MC")
    ax.set_xlabel("cos θ"); ax.set_ylabel("$N_i R_i^2$")
    ax.set_title(f"{run_label}\nsource = {source.tolist()}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle("Angular Response: Data vs MC")
plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_data_vs_mc.png", dpi=150)
plt.close()

# Combined scatter: all runs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors_run = {"R1767": "black", "R1769": "blue", "R2336": "green", "R2337": "red"}

for run_label, sig_path in runs.items():
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    
    # Data
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    N_i_data = pure_hits / n_cands
    A_data = N_i_data * R**2
    
    # MC
    qe_df = pd.read_parquet(mc_files[run_label])
    mc_hits = qe_df["mc_hits"].values
    N_i_mc = mc_hits / n_cands_mc[run_label]
    A_mc = N_i_mc * R**2
    
    valid_d = (N_i_data > 0) & (cos_theta > 0)
    valid_m = (N_i_mc > 0) & (cos_theta > 0)
    
    ax1.scatter(cos_theta[valid_d], A_data[valid_d],
                s=3, alpha=0.3, color=colors_run[run_label], label=f"{run_label}")
    ax2.scatter(cos_theta[valid_m], A_mc[valid_m],
                s=3, alpha=0.3, color=colors_run[run_label], label=f"{run_label}")

ax1.set_xlabel("cos θ"); ax1.set_ylabel("$N_i R_i^2$")
ax1.set_title("Data — all positions"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2.set_xlabel("cos θ"); ax2.set_ylabel("$N_i R_i^2$")
ax2.set_title("MC — all positions"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.suptitle("Angular Response: all source positions combined")
plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_combined_data_mc.png", dpi=150)
plt.close()

# 2D histograms: Data vs MC, In-situ vs Ex-situ
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
from matplotlib.colors import LogNorm

all_data = {"insitu": {"cos": [], "A": []}, "exsitu": {"cos": [], "A": []}}
all_mc   = {"insitu": {"cos": [], "A": []}, "exsitu": {"cos": [], "A": []}}

for run_label, sig_path in runs.items():
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    N_i_data = pure_hits / n_cands
    A_data = N_i_data * R**2
    
    qe_df = pd.read_parquet(mc_files[run_label])
    N_i_mc = qe_df["mc_hits"].values / n_cands_mc[run_label]
    A_mc = N_i_mc * R**2
    
    valid_d = (N_i_data > 0) & (cos_theta > 0)
    valid_m = (N_i_mc > 0) & (cos_theta > 0)
    
    for label, mask_type in [("insitu", is_insitu), ("exsitu", ~is_insitu)]:
        md = valid_d & mask_type
        mm = valid_m & mask_type
        all_data[label]["cos"].extend(cos_theta[md])
        all_data[label]["A"].extend(A_data[md])
        all_mc[label]["cos"].extend(cos_theta[mm])
        all_mc[label]["A"].extend(A_mc[mm])

titles = [("Data Ex-situ", "exsitu", all_data),
          ("Data In-situ", "insitu", all_data),
          ("MC Ex-situ",   "exsitu", all_mc),
          ("MC In-situ",   "insitu", all_mc)]

for ax, (title, key, source_dict) in zip(axes.flatten(), titles):
    c = np.array(source_dict[key]["cos"])
    a = np.array(source_dict[key]["A"])
    if len(c) > 0:
        ax.hist2d(c, a, bins=[20, 20], norm=LogNorm())
    ax.set_xlabel("cos θ"); ax.set_ylabel("$N_i R_i^2$")
    ax.set_title(title)

plt.suptitle("Angular Response 2D — all positions combined")
plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_2d_data_mc.png", dpi=150)
plt.close()

# ─── 9. Double ratio: Data ratio / MC ratio (should be 1) ───

# Choose pairs of runs to compare
run_pairs = [
    ("R1767", "R1769"),
    ("R1767", "R2336"),
    ("R1767", "R2337"),
    ("R1769", "R2336"),
    ("R1769", "R2337"),
    ("R2336", "R2337")
]

# n_cands_mc = {"R1767": 45818, "R1769": 49548, "R2336": 43446, "R2337": 43316}

fig, axes = plt.subplots(len(run_pairs), 2, figsize=(16, 5*len(run_pairs)))

for row, (run_a, run_b) in enumerate(run_pairs):
    
    # Data hits per PMT (pure signal)
    pure_a, n_a = get_pure_hits(runs[run_a], BKG, trms_cut=2.0)
    pure_b, n_b = get_pure_hits(runs[run_b], BKG, trms_cut=2.0)
    
    # MC hits per PMT
    qe_a = pd.read_parquet(mc_files[run_a])
    qe_b = pd.read_parquet(mc_files[run_b])
    mc_a = qe_a["mc_hits"].values
    mc_b = qe_b["mc_hits"].values
    
    # Normalize per candidate
    rate_data_a = pure_a / n_a
    rate_data_b = pure_b / n_b
    rate_mc_a   = mc_a / n_cands_mc[run_a]
    rate_mc_b   = mc_b / n_cands_mc[run_b]
    
    # Single ratios
    eta_data = np.zeros(N_PMTS)
    eta_mc   = np.zeros(N_PMTS)
    
    valid_data = (rate_data_a > 0) & (rate_data_b > 0)
    valid_mc   = (rate_mc_a > 0) & (rate_mc_b > 0)
    valid      = valid_data & valid_mc
    
    eta_data[valid] = rate_data_a[valid] / rate_data_b[valid]
    eta_mc[valid]   = rate_mc_a[valid]   / rate_mc_b[valid]
    
    # Double ratio
    double_ratio = np.zeros(N_PMTS)
    double_ratio[valid] = eta_data[valid] / eta_mc[valid]
    
    # Error propagation (Poisson)
    err_double = np.zeros(N_PMTS)
    err_double[valid] = double_ratio[valid] * np.sqrt(
        1/pure_a[valid] + 1/pure_b[valid] + 1/mc_a[valid] + 1/mc_b[valid]
    )
    
    # ─── Plot 1: double ratio per PMT ───
    ax1 = axes[row, 0]
    
    for cat, color in category_colors.items():
        mask = valid & (pmt_category == cat)
        if mask.sum() > 0:
            ax1.errorbar(np.where(mask)[0], double_ratio[mask], yerr=err_double[mask],
                         fmt="o", markersize=2, color=color, ecolor=color,
                         elinewidth=0.3, capsize=0, alpha=0.5, label=cat)
    
    ax1.axhline(1.0, linestyle=":", color="k")
    ax1.set_xlabel("PMT ID")
    ax1.set_ylabel("$\\eta_i / \\tilde{\\eta}_i$")
    ax1.set_title(f"Double ratio: {run_a}/{run_b}")
    ax1.set_ylim(0, 3)
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
    
    # ─── Plot 2: projection (histogram) ───
    ax2 = axes[row, 1]
    
    for cat, color in category_colors.items():
        mask = valid & (pmt_category == cat)
        vals = double_ratio[mask]
        if len(vals) > 0:
            vals_clipped = vals[(vals >= 0) & (vals <= 3)]
            ax2.hist(vals, bins=50, range=(0, 3), histtype="step",
                     color=color, linewidth=1.5,
                     label=f"{cat} (μ={vals_clipped.mean():.2f}, σ={vals_clipped.std():.2f})")
    
    ax2.axvline(1.0, linestyle=":", color="k")
    ax2.set_xlabel("$\\eta_i / \\tilde{\\eta}_i$")
    ax2.set_ylabel("PMTs")
    ax2.set_title(f"Double ratio distribution: {run_a}/{run_b}")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

plt.suptitle("Double Ratio Check: $\\eta_{data} / \\eta_{MC}$ (should be 1 if acceptance = MC)", 
             y=1.01, fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/double_ratio_check.png", dpi=150, bbox_inches='tight')
plt.close()

# ─── Summary stats ───
print("\n--- Double Ratio Summary ---")
for run_a, run_b in run_pairs:
    pure_a, n_a = get_pure_hits(runs[run_a], BKG, trms_cut=2.0)
    pure_b, n_b = get_pure_hits(runs[run_b], BKG, trms_cut=2.0)
    qe_a = pd.read_parquet(mc_files[run_a])
    qe_b = pd.read_parquet(mc_files[run_b])
    mc_a, mc_b = qe_a["mc_hits"].values, qe_b["mc_hits"].values
    
    rate_da, rate_db = pure_a/n_a, pure_b/n_b
    rate_ma, rate_mb = mc_a/n_cands_mc[run_a], mc_b/n_cands_mc[run_b]
    
    valid = (rate_da > 0) & (rate_db > 0) & (rate_ma > 0) & (rate_mb > 0)
    dr = (rate_da[valid]/rate_db[valid]) / (rate_ma[valid]/rate_mb[valid])
    
    print(f"  {run_a}/{run_b}: mean={dr.mean():.3f}, std={dr.std():.3f}, "
          f"median={np.median(dr):.3f}, N_PMTs={valid.sum()}")
    
# ─── 10. Angular acceptance: N_i R_i^2 / RQE_i ───
# Divide by relative QE to cancel per-PMT QE variations and isolate A(θ)

fig, axes = plt.subplots(1, len(runs), figsize=(6*len(runs), 6))
if len(runs) == 1:
    axes = [axes]

for ax, (run_label, sig_path) in zip(axes, runs.items()):
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    N_i = pure_hits / n_cands
    
    # Load RQE for this run
    qe_df = pd.read_parquet(mc_files[run_label])
    rqe = qe_df["relative_qe"].values
    
    # N_i * R^2 / RQE — should isolate angular acceptance
    A_theta_clean = np.zeros(N_PMTS)
    valid_rqe = rqe > 0
    A_theta_clean[valid_rqe] = (N_i[valid_rqe] * R[valid_rqe]**2) / rqe[valid_rqe]
    
    valid = (N_i > 0) & (cos_theta > 0) & valid_rqe
    
    for cat, color in category_colors.items():
        mask = valid & (pmt_category == cat)
        if mask.sum() > 0:
            ax.scatter(cos_theta[mask], A_theta_clean[mask],
                       s=3, alpha=0.5, color=color, label=cat)
    
    ax.set_xlabel("cos θ")
    ax.set_ylabel("$N_i R_i^2 / \\epsilon^Q_{rel}$")
    ax.set_title(f"{run_label} — Angular acceptance only\nsource = {source.tolist()}")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle("Pure angular acceptance: $N_i R_i^2 / \\epsilon^Q_{rel}$ vs cos θ")
plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_qe_corrected.png", dpi=150)
plt.close()

# Combined: all runs together
fig, ax = plt.subplots(figsize=(10, 7))

for run_label, sig_path in runs.items():
    source = source_positions[run_label]
    cos_theta, R = compute_angles(source)
    pure_hits, n_cands = get_pure_hits(sig_path, BKG, trms_cut=2.0)
    N_i = pure_hits / n_cands
    
    qe_df = pd.read_parquet(mc_files[run_label])
    rqe = qe_df["relative_qe"].values
    
    A_theta_clean = np.zeros(N_PMTS)
    valid_rqe = rqe > 0
    A_theta_clean[valid_rqe] = (N_i[valid_rqe] * R[valid_rqe]**2) / rqe[valid_rqe]
    
    valid = (N_i > 0) & (cos_theta > 0) & valid_rqe
    
    ax.scatter(cos_theta[valid], A_theta_clean[valid],
               s=3, alpha=0.4, color=colors_run[run_label], label=run_label)

ax.set_xlabel("cos θ")
ax.set_ylabel("$N_i R_i^2 / \\epsilon^Q_{rel}$")
ax.set_title("Angular acceptance (QE-corrected) — All source positions")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{BASE}/angular_response_qe_corrected_combined.png", dpi=150)
plt.close()

print("QE-corrected angular acceptance plots saved!")

print("Angular response Data vs MC plots done!")