#%%
import sys
import os
import uproot
import argparse
import gc
from copy import deepcopy
from collections import defaultdict

# Add HK Software Directory
sys.path.append(os.path.abspath("/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk"))
sys.path.append(os.path.abspath("/home/usc/ie/dcr/hk/nicf_analysis/data_analysis/src"))

import awkward as ak
import numpy   as np

from data_manipulation_functions          import remove_spill, spillnHits, candidatenHits
from WCTE_BRB_Data_Analysis.wcte.brbtools import sort_run_files, get_part_files
from nHits_trigger.src.read_data          import nHits

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run",       type=int,  required=True,  help="Run Number")
parser.add_argument("--parts",                required=True,  help="Parts to analyse (int or 'all')")
parser.add_argument("--chargeCut", type=bool, required=True,  help="Apply Q20 charge cut")
# vTc-specific tuneable parameters (all have sensible defaults matching the notebook)
parser.add_argument("--tc_cut",       type=float, default=5.0,   help="tc <= tc_cut [ns] (default: 5.0)")
parser.add_argument("--q20_min",      type=float, default=400.0, help="Q20 lower bound [ADC] (default: 400)")
parser.add_argument("--q20_max",      type=float, default=12500., help="Q20 upper bound [ADC] (default: 12500)")
parser.add_argument("--thresh_inf",   type=int,   default=5,     help="nHits lower threshold (default: 5)")
parser.add_argument("--thresh_sup",   type=int,   default=60,    help="nHits upper threshold (default: 60)")
parser.add_argument("--pre_window",   type=int,   default=200,   help="Pre-trigger window [ns] (default: 200)")
parser.add_argument("--post_window",  type=int,   default=200,   help="Post-trigger window [ns] (default: 200)")
args = parser.parse_args()

run        = args.run
parts      = str(args.parts)
use_charge = args.chargeCut

# vTc parameters
TC_CUT      = args.tc_cut
Q20_MIN     = args.q20_min
Q20_MAX     = args.q20_max
THRESH_INF  = args.thresh_inf
THRESH_SUP  = args.thresh_sup
PRE_WINDOW  = args.pre_window
POST_WINDOW = args.post_window
SLIDING_W   = 20   # ns — fixed, same as notebook
DEAD_TIME   = PRE_WINDOW + POST_WINDOW

# Output file (same naming convention as the original, drop-in replacement)
min_charge = int(Q20_MIN)
max_charge = int(Q20_MAX)
bonsai_output_file = (
    f"/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/nicf_data/data/"
    f"run_{run}_forBONSAI_separated_chargeFiltered{min_charge}-{max_charge}_vTc.csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print("+----------------------------------------------+")
print(f"+                   Run {run}                   +")
print("+  Data Manipulation And Filtering Started!    +")
print("+        vTc pipeline (notebook version)       +")
print(f"+  tc_cut={TC_CUT} ns  Q20=[{Q20_MIN},{Q20_MAX}]        +")
print("+----------------------------------------------+")

# ─────────────────────────────────────────────────────────────────────────────
# READ RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
run_files  = sort_run_files(
    f"/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/raw_data/production_v0/{run}/"
    f"WCTE_offline_R{run}S*P*.root"
)
part_files = get_part_files(run_files)
if parts != "all":
    part_files = part_files[:int(parts)]

run_corrected_times           = []
run_hit_charges               = []
run_hit_card_ids              = []
run_hit_channel_ids           = []
run_hit_slot_ids              = []
run_hit_position_ids          = []
run_hit_pmt_has_time_constant = []

for part in part_files:
    print(f"Processing part WCTE_offline_R{run}S0P{part}")
    tree = uproot.open(run_files[part] + ":WCTEReadoutWindows")

    file_hit_card_ids              = ak.values_astype(tree["hit_mpmt_card_ids"]        .array(), np.int16)
    file_hit_channel_ids           = ak.values_astype(tree["hit_pmt_channel_ids"]      .array(), np.int8)
    file_hit_times_calib           = ak.values_astype(tree["hit_pmt_calibrated_times"] .array(), np.float64)
    file_hit_charges               = ak.values_astype(tree["hit_pmt_charges"]          .array(), np.float64)
    file_hit_slot_ids              = ak.values_astype(tree["hit_mpmt_slot_ids"]        .array(), np.int16)
    file_hit_position_ids          = ak.values_astype(tree["hit_pmt_position_ids"]     .array(), np.int16)
    file_hit_pmt_has_time_constant = ak.values_astype(tree["hit_pmt_has_time_constant"].array(), np.bool_)

    mask = (
        (file_hit_charges < 1e4) &
        (file_hit_card_ids < 120) &
        (file_hit_pmt_has_time_constant != 0)
    )
    corrected_times                = file_hit_times_calib           [mask]
    file_hit_charges               = file_hit_charges               [mask]
    file_hit_card_ids              = file_hit_card_ids              [mask]
    file_hit_channel_ids           = file_hit_channel_ids           [mask]
    file_hit_slot_ids              = file_hit_slot_ids              [mask]
    file_hit_position_ids          = file_hit_position_ids          [mask]
    file_hit_pmt_has_time_constant = file_hit_pmt_has_time_constant [mask]

    order = ak.argsort(corrected_times)

    run_corrected_times          .append(corrected_times               [order])
    run_hit_charges              .append(file_hit_charges              [order])
    run_hit_card_ids             .append(file_hit_card_ids             [order])
    run_hit_channel_ids          .append(file_hit_channel_ids          [order])
    run_hit_slot_ids             .append(file_hit_slot_ids             [order])
    run_hit_position_ids         .append(file_hit_position_ids         [order])
    run_hit_pmt_has_time_constant.append(file_hit_pmt_has_time_constant[order])

run_corrected_times           = ak.concatenate(run_corrected_times          )
run_hit_charges               = ak.concatenate(run_hit_charges              )
run_hit_card_ids              = ak.concatenate(run_hit_card_ids             )
run_hit_channel_ids           = ak.concatenate(run_hit_channel_ids          )
run_hit_slot_ids              = ak.concatenate(run_hit_slot_ids             )
run_hit_position_ids          = ak.concatenate(run_hit_position_ids         )
run_hit_pmt_has_time_constant = ak.concatenate(run_hit_pmt_has_time_constant)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SPILL REMOVAL
# ─────────────────────────────────────────────────────────────────────────────
print("Step 1: Spill removal...")
triggered_spill_hits_index, _ = spillnHits(
    mode="multiple_events", hit_times=run_corrected_times,
    w=5000, thresh_min=300, thresh_max=100000,
    pre_window=0, post_window=4000, jump=9000
)
noSpill_times = remove_spill(triggered_spill_hits_index, run_corrected_times)

# Rebuild charge/slot/position arrays aligned with noSpill_times
# (remove_spill returns a new ak.Array with the same event structure but
#  spill hits dropped; we replicate the same index removal for the other vars)
def remove_spill_var(trigger_indices, run_var):
    """Apply the same spill-index removal to any per-event variable."""
    out = []
    for i in tqdm(range(len(run_var)), leave=False):
        if len(trigger_indices[i]) > 0:
            data  = run_var[i]
            tidx  = np.concatenate(trigger_indices[i])
            valid = np.setdiff1d(np.arange(len(data)), tidx)
            out.append(data[valid])
        else:
            out.append(run_var[i])
    return ak.Array(out)

noSpill_charges     = remove_spill_var(triggered_spill_hits_index, run_hit_charges)
noSpill_slot_ids    = remove_spill_var(triggered_spill_hits_index, run_hit_slot_ids)
noSpill_position_ids= remove_spill_var(triggered_spill_hits_index, run_hit_position_ids)
noSpill_card_ids    = remove_spill_var(triggered_spill_hits_index, run_hit_card_ids)
noSpill_channel_ids = remove_spill_var(triggered_spill_hits_index, run_hit_channel_ids)

del triggered_spill_hits_index, run_corrected_times
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SIGNAL TRIGGER  (20 ns sliding window, ±pre/post extension)
# ─────────────────────────────────────────────────────────────────────────────
print("Step 2: Signal trigger (nHits sliding window)...")
triggered_hits_index, triggered_hit_times, d_seed, _ = candidatenHits(
    mode="multiple_events", hit_times=noSpill_times,
    w=SLIDING_W, thresh_min=THRESH_INF, thresh_max=THRESH_SUP,
    pre_window=PRE_WINDOW, post_window=POST_WINDOW,
    jump=DEAD_TIME
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DOUBLE-TRIGGER REMOVAL
# ─────────────────────────────────────────────────────────────────────────────
print("Step 3: Double-trigger removal...")
double_candidates = {}

for k, v in tqdm(triggered_hit_times.items(), total=len(triggered_hit_times)):
    sett = triggered_hit_times[k]
    seed = d_seed[k]
    for c in range(len(v)):
        candidate_times = [sett[c] - seed[c]]
        test_idx, test_times, _, _ = candidatenHits(
            mode="single_event", hit_times=candidate_times,
            w=SLIDING_W, thresh_min=THRESH_INF, thresh_max=np.inf,
            pre_window=0, post_window=0, jump=SLIDING_W,
            event=0, progress_bar=False
        )
        if len(test_times[0]) > 1:
            double_candidates[k] = c

# Remove double-trigger candidates
triggered_hit_times_clean  = deepcopy(triggered_hit_times)
triggered_hits_index_clean = deepcopy(triggered_hits_index)
d_seed_clean               = deepcopy(d_seed)

to_delete = defaultdict(list)
for k, idx in double_candidates.items():
    to_delete[k].append(idx)

for k, indices in to_delete.items():
    for idx in sorted(indices, reverse=True):
        if k in triggered_hit_times_clean and idx < len(triggered_hit_times_clean[k]):
            del triggered_hit_times_clean [k][idx]
            del triggered_hits_index_clean[k][idx]
            del d_seed_clean              [k][idx]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SHIFT TIMES TO SEED  +  tc CUT
# ─────────────────────────────────────────────────────────────────────────────
print(f"Step 4: Shifting times and applying tc <= {TC_CUT} ns cut...")

def compute_tc_20(shifted_times):
    mask = (shifted_times >= 0) & (shifted_times <= 20)
    if np.any(mask):
        return np.mean(shifted_times[mask])
    return None

triggered_hit_times_clean_shifted_tc  = {}
triggered_hits_index_clean_tc         = {}

for k, arrays in triggered_hit_times_clean.items():
    if k not in triggered_hits_index_clean:
        continue
    new_times   = []
    new_indices = []
    for i, arr in enumerate(arrays):
        shifted = arr - d_seed_clean[k][i]
        tc20    = compute_tc_20(shifted)
        if tc20 is not None and tc20 <= TC_CUT:
            new_times  .append(shifted)
            new_indices.append(triggered_hits_index_clean[k][i])
    triggered_hit_times_clean_shifted_tc [k] = new_times
    triggered_hits_index_clean_tc        [k] = new_indices

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Q20 CUT  (only if --chargeCut True)
# ─────────────────────────────────────────────────────────────────────────────
triggered_hit_times_final  = {}
triggered_hits_index_final = {}

if use_charge:
    print(f"Step 5: Q20 charge cut [{Q20_MIN}, {Q20_MAX}] ADC...")
    for k, arrays in triggered_hit_times_clean_shifted_tc.items():
        if k not in triggered_hits_index_clean_tc:
            continue
        new_times   = []
        new_indices = []
        for i, arr in enumerate(arrays):
            mask_20        = (arr >= 0) & (arr <= 20)
            hit_indices_20 = triggered_hits_index_clean_tc[k][i][mask_20]
            charges_20     = ak.to_numpy(noSpill_charges[k][hit_indices_20])
            q20            = np.sum(charges_20)
            if Q20_MIN <= q20 <= Q20_MAX:
                new_times  .append(arr)
                new_indices.append(triggered_hits_index_clean_tc[k][i])
        triggered_hit_times_final [k] = new_times
        triggered_hits_index_final[k] = new_indices
else:
    print("Step 5: Skipping Q20 charge cut.")
    triggered_hit_times_final  = triggered_hit_times_clean_shifted_tc
    triggered_hits_index_final = triggered_hits_index_clean_tc

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — BUILD OUTPUT: only 20 ns window hits per candidate
# ─────────────────────────────────────────────────────────────────────────────
print("Step 6: Building output arrays (20 ns window only)...")

times_out        = []
charges_out      = []
card_ids_out     = []
slot_ids_out     = []
channel_ids_out  = []
position_ids_out = []
events_out       = []

event_counter = 0

for k, arrays in tqdm(triggered_hit_times_final.items(), total=len(triggered_hit_times_final)):
    for i, shifted_arr in enumerate(arrays):
        hits       = triggered_hits_index_final[k][i]
        mask_20    = (shifted_arr >= 0) & (shifted_arr <= 20)
        hits_20    = hits[mask_20]

        if len(hits_20) == 0:
            continue

        n = len(hits_20)
        times_out       .append(ak.to_numpy(noSpill_times         [k][hits_20]))
        charges_out     .append(ak.to_numpy(noSpill_charges       [k][hits_20]))
        card_ids_out    .append(ak.to_numpy(noSpill_card_ids      [k][hits_20]))
        slot_ids_out    .append(ak.to_numpy(noSpill_slot_ids      [k][hits_20]))
        channel_ids_out .append(ak.to_numpy(noSpill_channel_ids   [k][hits_20]))
        position_ids_out.append(ak.to_numpy(noSpill_position_ids  [k][hits_20]))
        events_out      .append(np.full(n, event_counter, dtype=int))
        event_counter += 1

times        = np.concatenate(times_out)
charges      = np.concatenate(charges_out)
card_ids     = np.concatenate(card_ids_out)
slot_ids     = np.concatenate(slot_ids_out)
channel_ids  = np.concatenate(channel_ids_out)
position_ids = np.concatenate(position_ids_out)
events       = np.concatenate(events_out)

data = np.column_stack([
    events,
    times,
    charges,
    card_ids,
    slot_ids,
    channel_ids,
    position_ids
])

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
np.savetxt(
    bonsai_output_file,
    data,
    delimiter=",",
    header="event_id,hit_pmt_calibrated_times,hit_pmt_charges,hit_mpmt_card_ids,hit_mpmt_slot_ids,hit_pmt_channel_ids,hit_pmt_position_ids",
    comments="",
    fmt=["%d", "%.8f", "%d", "%d", "%d", "%d", "%d"]
)

print(f"Total hits saved: {len(data)}  |  Total candidates: {event_counter}")
print(f"Output: {bonsai_output_file}")

print("+---------------------------------------+")
print(f"+              Run {run}                 +")
print("+ Data Manipulation And Filtering Done! +")
print("+      (vTc pipeline)  → BONSAI         +")
print("+---------------------------------------+")
