
import ROOT 
import glob 
import numpy as np 
import pandas as pd 
import sys 
import array
import pandas as pd
import os
import cppyy

cppyy.add_include_path("/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/include")
cppyy.load_library("/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/lib/libWCSimRoot.so")

cppyy.add_include_path("/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/hk-BONSAI/bonsai")
cppyy.load_library("/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/hk-BONSAI/libWCSimBonsai.so")

def get_offline_run_files(run, base="/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/raw_data"):
    files = glob.glob(f"{base}/{run}/WCTE_offline_R*.root")
    return files

def get_offline_run_tchain(run, limit=0):
    files = get_offline_run_files(run)
    tt = ROOT.TChain("WCTEReadoutWindows")
    count = 0
    for ff in files:
        tt.Add(ff)
        count += 1
        if limit > 0 and count > limit: break
    return tt

# Function to load geometry mapping from a text file
def get_geo_mapping():
    geo = pd.read_csv("/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/build/mydir/geofile_NuPRISMBeamTest_16cShort_mPMT.txt", 
                      index_col=False,      # Do not use any column as the index
                      sep='\s+',            # Use whitespace as separator
                      skiprows=5,           # Skip the first 5 header lines
                      names=["id","mpmtid","spmtid",
                             "x","y","z","dx","dy","dz", "cyloc"])  # Explicit column names

    return geo

geo = get_geo_mapping()

def getxyz(geo, mpmtids, posids):
    # Build a single lookup dictionary: {(mpmtid, spmtid): (x, y, z, id)}
    lookup = {
        (row.mpmtid, row.spmtid): (row.x, row.y, row.z, row.id)
        for row in geo.itertuples(index=False)
    }

    # Adjust input IDs to match geometry convention
    keys = zip((mid + 1 for mid in mpmtids), (sid + 1 for sid in posids))

    # Use the lookup dictionary to retrieve values efficiently
    results = [lookup.get(k, (-999.9, -999.9, -999.9, -999)) for k in keys]

    # Unpack results into separate arrays
    if len(results) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    x, y, z, c = map(np.array, zip(*results))

    return x, y, z, c


# Simple class to store a filtered time slice of hit data
class hit_slice:
    def __init__(self, hits):
        self.t = hits.t  # Hit times
        self.q = hits.q  # Hit charges
        self.x = hits.x  # x-positions
        self.y = hits.y  # y-positions
        self.z = hits.z  # z-positions
        self.cable = hits.cable 

# Class to process and store a collection of hits from an event
class hit_collection:
    def __init__(self, e, g=geo):
        # If 'e' is a DataFrame, extract directly the columns
        if isinstance(e, pd.DataFrame):
            df = e
            # Assume we already time constant filtered
            self.t       = df["hit_pmt_calibrated_times"].values
            self.q       = df["hit_pmt_charges"].values
            self.slot    = df["hit_mpmt_slot_ids"].values
            self.posid   = df["hit_pmt_position_ids"].values

            # Geometric Map
            self.x, self.y, self.z, self.cable = getxyz(g, self.slot, self.posid)

            # # Cut in geom
            # mask = self.x > -999
            # for attr in ("t", "q", "card", "slot", "channel", "posid", "x", "y", "z", "cable"):
            #     setattr(self, attr, getattr(self, attr)[mask])

            # Remove hits with invalid geometry mapping (sentinel x = -999.9)
            cut          = self.x > -999
            self.t       = self.t      [cut]
            self.q       = self.q      [cut]
            self.slot    = self.slot   [cut]
            self.posid   = self.posid  [cut]
            self.z       = self.z      [cut]
            self.y       = self.y      [cut]
            self.cable   = self.cable  [cut]
            self.x       = self.x      [cut]
            
        
        else:
            # Apply time window cut for hit calibration times
            above_10us  = np.array(e.hit_pmt_calibrated_times) > 10000
            below_490us = np.array(e.hit_pmt_calibrated_times) < 490000
            has_calib   = np.array(e.hit_pmt_has_time_constant)  # Check for valid calibration
            hit_selection = has_calib # Combined mask

            # Apply selection mask to extract relevant hit information
            self.t       = np.array(e.hit_pmt_calibrated_times) [hit_selection]
            self.q       = np.array(e.hit_pmt_charges)          [hit_selection]
            self.slot    = np.array(e.hit_mpmt_slot_ids)        [hit_selection]
            self.posid   = np.array(e.hit_pmt_position_ids)     [hit_selection]

            # Get spatial positions and cable IDs using geometry mapping
            self.x, self.y, self.z, self.cable = getxyz(g, self.slot, self.posid)

            # Remove hits with invalid geometry mapping (sentinel x = -999.9)
            cut          = self.x > -999
            self.t       = self.t      [cut]
            self.q       = self.q      [cut]
            self.slot    = self.slot   [cut]
            self.posid   = self.posid  [cut]
            self.z       = self.z      [cut]
            self.y       = self.y      [cut]
            self.cable   = self.cable  [cut]
            self.x       = self.x      [cut]

    # Return a new hit_slice object within a time window
    def time_slice(self, tstart, tend):
        h    = hit_slice(self)                                 # Create a new slice from this collection
        tsel = (h.cable > -1) & (h.t > tstart) & (h.t < tend)  # Select hits within the time range

        # Apply selection to all relevant hit attributes
        h.q     = h.q[ tsel ]
        h.x     = h.x[ tsel ]
        h.y     = h.y[ tsel ]
        h.z     = h.z[ tsel ]
        h.cable = h.cable[ tsel ]
        h.t     = h.t[ tsel ]      

        return h  # Return the filtered slice
    
# Load the geometry mapping into a DataFrame
runno = int(sys.argv[1])
# ROOT Input
# tt    = get_offline_run_tchain(runno)

# DF Input
data   = pd.read_csv("../df_unfiltered_1767.csv", index_col=0)
events = np.unique(data["event_id"])
    
# Setup HKBONSAI with WCTE geo
simfile = ROOT.TFile("bonsai_reqs/wcsim.root")
simtree = simfile.Get("wcsimGeoT")

geotree = None
for geoevent in simtree:
    geotree = geoevent.wcsimrootgeom
    break
    
bonsai = cppyy.gbl.WCSimBonsai()
bonsai.Init(geotree)

# Start Filter
ns = 1
count = 0
nhit_cut = 10
mvwindow_start =   10000*ns
mvwindow_end   = 5000000*ns
mvwindow_step  =      10*ns
mvwindow_width =      10*ns

vertex = {
"nhits": [],
"nhitso": [],
"x": [],
"y": [],
"z": [],
"result0": [],
"result1": [],
"result2": [],
"result3": [],
"result4": [],
"result5": [],
"good0":[],
"good1":[],
"good2":[] 
}

for df in events:
    count += 1    
    if count > 1000: break
    print(count)

    # Generate hit_collection from event
    sub = data[data.event_id == df]
    hits_df = hit_collection(sub)

    # Run a moving average 50ns hit filter
    tstart = mvwindow_start 
    tend   = tstart + mvwindow_width
    tmax   = np.max(hits_df.t)
    
    while tend < tmax:
        
        window = hits_df.time_slice(tstart, tend)
        tstart += mvwindow_step
        tend   = tstart + mvwindow_width
    
        # Run prehit filter for this window
        if len(window.t) < nhit_cut: continue
        if len(window.t) > 400: continue
                
        # Run Bonsai
        bsVertex = array.array('f',3*[0.0])
        bsResult = array.array('f',6*[0.0])
        bsGood = array.array('f',3*[0.0])
        bsNhit = array.array('i',[len(window.cable)])
        bsNsel = array.array('i',[0])

        # Generate hit collection for this triggger
        bsCAB_a = array.array('i', window.cable)
        bsT_a = array.array('f', window.t - np.min(window.t) + 200)
        bsQ_a = array.array('f', window.q)

        # Run Bonsai
        try:
            nhits = bonsai.BonsaiFit(bsVertex, bsResult, bsGood, bsNsel, bsNhit, bsCAB_a, bsT_a, bsQ_a);
        except:
            print("BONSAIFAILED");
            pass
        # print(nhits, bsVertex, bsResult, bsGood, bsNsel, bsNhit)

        vertex["nhits"].append(nhits)
        vertex["nhitso"].append(len(window.t))
        
        vertex["x"].append(bsVertex[0])
        vertex["y"].append(bsVertex[1])
        vertex["z"].append(bsVertex[2])
        vertex["result0"].append(bsResult[0])
        vertex["result1"].append(bsResult[1])
        vertex["result2"].append(bsResult[2])
        vertex["result3"].append(bsResult[3])
        vertex["result4"].append(bsResult[4])
        vertex["result5"].append(bsResult[5])
        vertex["good0"].append(bsGood[0])
        vertex["good1"].append(bsGood[1])
        vertex["good2"].append(bsGood[2])
        
        # Skip to next window if we found a hit
        tstart += mvwindow_width
        tend    = tstart + mvwindow_width
        
for key in vertex:
    print(key, len(vertex[key]))
    
df = pd.DataFrame(vertex)
df.to_csv(f"df_noSpill_vertex_{runno}.csv")
