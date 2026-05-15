import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np
import matplotlib.colors as colors

import matplotlib
import sys


sys.path.append('/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk')

# Import Diego's tools
from WCSimFilePackages.npz_to_df     import simple_truehits_info_to_df, truehits_info_to_df, digihits_info_to_df
from WCSimFilePackages.npz_to_df     import super_simple_track_info_to_df
from WCTE_event_display.EventDisplay import EventDisplay

from tqdm.notebook import tqdm

import hipy.hipy.pltext as pltext
import hipy.hipy.utils  as ut

pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)

pltext.style()

npz_1767 = '/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/npz/1Mneutrons_NiCf_piFix_QGSP_BIC_HP_pos1767_CDSON_15kevents.npz'
nevents  = 15000

# Creación del DataFrame de DigiHits usando la función digihits_info_to_df
df_trueHits_1767 = simple_truehits_info_to_df(npz_1767).dropna()
df_simpleTracks_1767 = super_simple_track_info_to_df(npz_1767).dropna()

# -------------------------------------------------------
# parámetro de geometría
# -------------------------------------------------------

R_source = 6.75   # radio de la esfera NiO (ajústalo a tu geometría)


# -------------------------------------------------------
# 1. seleccionar gammas
# -------------------------------------------------------

gammas = df_simpleTracks_1767[df_simpleTracks_1767.track_pid == 22].copy()


# -------------------------------------------------------
# 2. convertir coordenadas a float (soluciona tu error)
# -------------------------------------------------------

coords = gammas[["track_xi","track_yi","track_zi"]].astype("float32").values


# -------------------------------------------------------
# 3. calcular radio de creación de la gamma
# -------------------------------------------------------

gammas["r"] = np.sqrt((coords**2).sum(axis=1))


# -------------------------------------------------------
# 4. clasificar captura
# -------------------------------------------------------

gammas["capture_type"] = np.where(
    gammas["r"] < R_source,
    "nickel",
    "water"
)

gamma_labels = gammas[[
    "event_id",
    "track_id",
    "capture_type"
]].rename(columns={"track_id": "gamma_track_id"})


# -------------------------------------------------------
# 5. preparar árbol de tracks
# -------------------------------------------------------

tracks = df_simpleTracks_1767[[
    "event_id",
    "track_id",
    "track_parent"
]].copy()


# -------------------------------------------------------
# 6. asociar tracks hijos de las gammas
# -------------------------------------------------------

tracks = tracks.merge(
    gamma_labels,
    left_on=["event_id", "track_parent"],
    right_on=["event_id", "gamma_track_id"],
    how="left"
)

tracks = tracks.drop(columns=["gamma_track_id"])


# -------------------------------------------------------
# 7. propagar etiqueta a descendientes
# -------------------------------------------------------

for _ in range(5):

    parent_labels = tracks[[
        "event_id",
        "track_id",
        "capture_type"
    ]].rename(columns={
        "track_id":"parent_id",
        "capture_type":"parent_capture"
    })

    tracks = tracks.merge(
        parent_labels,
        left_on=["event_id","track_parent"],
        right_on=["event_id","parent_id"],
        how="left"
    )

    tracks["capture_type"] = tracks["capture_type"].fillna(
        tracks["parent_capture"]
    )

    tracks = tracks.drop(columns=["parent_id","parent_capture"])


track_labels = tracks[[
    "event_id",
    "track_id",
    "capture_type"
]]


# -------------------------------------------------------
# 8. etiquetar hits
# -------------------------------------------------------

hits = df_trueHits_1767.merge(
    track_labels,
    left_on=["event_id","true_hit_parent"],
    right_on=["event_id","track_id"],
    how="left"
)


# -------------------------------------------------------
# 9. separar muestras
# -------------------------------------------------------

hits_nickel = hits[hits.capture_type == "nickel"].copy()
hits_water  = hits[hits.capture_type == "water"].copy()


# -------------------------------------------------------
# resultado
# -------------------------------------------------------

print("Hits captura en Ni:", len(hits_nickel))
print("Hits captura en agua:", len(hits_water))