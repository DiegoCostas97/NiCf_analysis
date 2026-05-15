#%%
import sys
import os
import uproot
import argparse
import gc

# Add HK Software Directory
sys.path.append(os.path.abspath("/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk"))
sys.path.append(os.path.abspath("/home/usc/ie/dcr/hk/nicf_analysis/data_analysis/src"))

import hipy.hipy.pltext  as pltext
import matplotlib.pyplot as plt
import awkward           as ak
import numpy             as np

from data_manipulation_functions          import remove_spill, nHits, select_signal_clusters, filter_charge
from WCTE_BRB_Data_Analysis.wcte.brbtools import sort_run_files, get_part_files

from tqdm import tqdm

pltext.style()

# PARSER AND VARIABLES
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, required=True, help="Run Number")
parser.add_argument("--parts", required=True, help="Parts to analize")
parser.add_argument("--chargeCut", type=bool, required=True, help="Use charge cut or not (Note not using it will increase computation time by A LOT)")
args = parser.parse_args()

run = args.run
parts = str(args.parts)
charge = args.chargeCut

# READ RAW DATA
run_files  = sort_run_files(f"/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/raw_data/production_v0/{run}/WCTE_offline_R{run}S*P*.root")
part_files = get_part_files(run_files)
if parts == "all":
    part_files = part_files

else:
     part_files = part_files[:int(parts)]

if charge:
    print("+------------------------------------------+")
    print(f"+                Run {run}                 +")
    print("+ Data Manipulation And Filtering Started! +")
    print("+      Using Charge Cut Configuration      +")
    print("+------------------------------------------+")

else:
    print("+------------------------------------------+")
    print(f"+                Run {run}                 +")
    print("+ Data Manipulation And Filtering Started! +")
    print("+------------------------------------------+")

#%%
# Run Variables
run_corrected_times           = []
run_hit_charges               = []
run_hit_card_ids              = []
run_hit_channel_ids           = []
run_hit_slot_ids              = []
run_hit_position_ids          = []
run_hit_pmt_has_time_constant = []

ev_offset = 0
for part in part_files:
    print(f"Processing part WCTE_offline_R{run}S0P{part}")
    tree = uproot.open(run_files[part] + ":WCTEReadoutWindows")
    
    # Read the variables
    file_hit_card_ids              = ak.values_astype(tree["hit_mpmt_card_ids"]        .array(), np.int16)
    file_hit_channel_ids           = ak.values_astype(tree["hit_pmt_channel_ids"]      .array(), np.int8)
    file_hit_times                 = ak.values_astype(tree["hit_pmt_times"]            .array(), np.float64)
    file_hit_times_calib           = ak.values_astype(tree["hit_pmt_calibrated_times"] .array(), np.float64)
    file_hit_charges               = ak.values_astype(tree["hit_pmt_charges"]          .array(), np.float64)
    file_hit_slot_ids              = ak.values_astype(tree["hit_mpmt_slot_ids"]        .array(), np.int16)
    file_hit_position_ids          = ak.values_astype(tree["hit_pmt_position_ids"]     .array(), np.int16)
    file_hit_pmt_has_time_constant = ak.values_astype(tree["hit_pmt_has_time_constant"].array(), np.bool) 

    # Filter Out Main Things
    mask = (file_hit_charges < 1e4) & (file_hit_card_ids < 120) & (file_hit_pmt_has_time_constant != 0)
    corrected_times                = file_hit_times_calib           [mask]
    file_hit_charges               = file_hit_charges               [mask]
    file_hit_card_ids              = file_hit_card_ids              [mask]
    file_hit_channel_ids           = file_hit_channel_ids           [mask]
    file_hit_slot_ids              = file_hit_slot_ids              [mask]
    file_hit_position_ids          = file_hit_position_ids          [mask]
    file_hit_pmt_has_time_constant = file_hit_pmt_has_time_constant [mask]


    # SUPER IMPORTANT! We want the hit time time-ordered, and we want to order the other variables the same way so later we can filter out by index in the array
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

# SELECT AND FILTER SPILLS
triggered_spill_hits_index, triggered_spill_hit_times = nHits(mode="multiple_events", hit_times=run_corrected_times, w=5000, thresh_min=300, thresh_max=10000, pre_window=0, post_window=4000, jump=9000)
noSpill_times                                         = remove_spill(triggered_spill_hits_index, run_corrected_times)

del triggered_spill_hits_index
del run_corrected_times
gc.collect()

# SELECT DATA AND CREATE CONTROL PLOTS
thresh_inf     = 2   # hits
thresh_sup     = 140 # hits
sliding_window = 20  # ns
dead_time      = 20  # ns

# nHits
triggered_signal_hits_index, triggered_signal_hit_times = nHits(mode="multiple_events", hit_times=noSpill_times, w=sliding_window, thresh_min=thresh_inf, thresh_max=thresh_sup, pre_window=0, post_window=0, jump=dead_time)

# tRMS
# rms_50ns   = []
# a50ns_size = []

# for k,v in tqdm(triggered_signal_hit_times.items(), total=len(triggered_signal_hit_times.items()), leave=False):
#         for i in triggered_signal_hit_times.get(k):
#             rms_50ns.append(np.std(i, ddof=0))
#             a50ns_size.append(len(i))

# # Plot
# pltext.hist(a50ns_size, 140, range=(0, 140), ylog=True, xylabels="N20 [hits]", label="NiCf");
# plt.savefig(f"/home/usc/ie/dcr/hk/nicf_analysis/data_analysis/figures/test_nhits_{run}.pdf")
# plt.close()

# pltext.hist(rms_50ns, 100, ylog=False, xylabels="$t_{RMS}$ [ns]", label="NiCf");
# plt.savefig(f"/home/usc/ie/dcr/hk/nicf_analysis/data_analysis/figures/test_trms_{run}.pdf")

#%%
# SELECT SIGNAL HITS AMONG DATA
signal_indices, len_signal_clusters = select_signal_clusters(triggered_signal_hit_times, triggered_signal_hits_index, trms_cut_inf=0, trms_cut_sup=6.0, noHits_cut_inf=10, noHits_cut_sup=60)

#%%
times_df            = noSpill_times       [signal_indices]
charge_df           = run_hit_charges     [signal_indices]
hit_card_ids_df     = run_hit_card_ids    [signal_indices]
hit_channel_ids_df  = run_hit_channel_ids [signal_indices]
hit_slot_ids_df     = run_hit_slot_ids    [signal_indices]
hit_position_ids_df = run_hit_position_ids[signal_indices]

flat_times_df            = ak.flatten(times_df           , axis=1)
flat_charge_df           = ak.flatten(charge_df          , axis=1)
flat_hit_card_ids_df     = ak.flatten(hit_card_ids_df    , axis=1)
flat_hit_channel_ids_df  = ak.flatten(hit_channel_ids_df , axis=1)
flat_hit_slot_ids_df     = ak.flatten(hit_slot_ids_df    , axis=1)
flat_hit_position_ids_df = ak.flatten(hit_position_ids_df, axis=1)

flat_counts   = ak.flatten(len_signal_clusters)

unf_times_df            = ak.unflatten(flat_times_df,            flat_counts)
unf_charge_df           = ak.unflatten(flat_charge_df,           flat_counts)
unf_hit_card_ids_df     = ak.unflatten(flat_hit_card_ids_df,     flat_counts)
unf_hit_channel_ids_df  = ak.unflatten(flat_hit_channel_ids_df,  flat_counts)
unf_hit_slot_ids_df     = ak.unflatten(flat_hit_slot_ids_df,     flat_counts)
unf_hit_position_ids_df = ak.unflatten(flat_hit_position_ids_df, flat_counts)

events           = np.arange(0, len(unf_charge_df))
n_hits_per_event = ak.num(unf_charge_df)
events           = np.repeat(events, n_hits_per_event)

events       = np.repeat(np.arange(len(unf_charge_df)), ak.num(unf_charge_df))
times        = ak.to_numpy(ak.ravel(unf_times_df))
charges      = ak.to_numpy(ak.ravel(unf_charge_df))
card_ids     = ak.to_numpy(ak.ravel(unf_hit_card_ids_df))
slot_ids     = ak.to_numpy(ak.ravel(unf_hit_slot_ids_df))
channel_ids  = ak.to_numpy(ak.ravel(unf_hit_channel_ids_df))
position_ids = ak.to_numpy(ak.ravel(unf_hit_position_ids_df))

# Combine Columns
data = np.column_stack([
    events.astype(int),
    times,
    charges,
    card_ids,
    slot_ids,
    channel_ids,
    position_ids
])

# Charge Cut
min_charge = 500
max_charge = 12500
bonsai_output_file = f"/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/nicf_data/data/run_{run}_forBONSAI_separated_chargeFiltered{min_charge}-{max_charge}.csv"

if charge: # We can apply a third cut using charge (see NB for nicf/bkg charge distribution)
    charge_filtered_data = filter_charge(events, charges, data, min_charge, max_charge) # Summed per event

    # Save CSV
    np.savetxt(
        bonsai_output_file,
        charge_filtered_data, # charge_filtered_data if you want charge-filtered data 
        delimiter=",",
        header="event_id,hit_pmt_calibrated_times,hit_pmt_charges,hit_mpmt_card_ids,hit_mpmt_slot_ids,hit_pmt_channel_ids,hit_pmt_position_ids",
        comments="",
        fmt=["%d", "%.8f", "%d", "%d", "%d", "%d", "%d"]
    )
    
else:
    # Save CSV
    np.savetxt(
        bonsai_output_file,
        data, 
        delimiter=",",
        header="event_id,hit_pmt_calibrated_times,hit_pmt_charges,hit_mpmt_card_ids,hit_mpmt_slot_ids,hit_pmt_channel_ids,hit_pmt_position_ids",
        comments="",
        fmt=["%d", "%.8f", "%d", "%d", "%d", "%d", "%d"]
    )

print(f"Total hits: {len(data)}, selected hits: {len(charge_filtered_data)}")

print("+---------------------------------------+")
print(f"+                Run {run}             +")
print("+ Data Manipulation And Filtering Done! +")
print("+        Let's Start With BONSAI        +")
print("+---------------------------------------+")
