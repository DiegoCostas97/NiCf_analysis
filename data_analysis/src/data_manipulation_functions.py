from tqdm import tqdm

import numpy   as np
import awkward as ak

def remove_spill(trigger_indices, run_times):
    corrected_run_times = []

    for i in tqdm(range(len(run_times))):
        #Event with spill → remove selected indices
        if len(trigger_indices[i]) > 0:
            data = run_times[i]
            trigger_indices_event = np.concatenate(trigger_indices[i])
            all_indices           = np.arange(len(data))
            valid_indices         = np.setdiff1d(all_indices, trigger_indices_event)
            corrected_run_times.append(data[valid_indices])
            
        else:
            # Event without spill → leave as it is
            corrected_run_times.append(run_times[i])      

    return ak.Array(corrected_run_times)

def spillnHits(mode, hit_times, w, thresh_min, thresh_max, pre_window, post_window, jump, event=0, progress_bar=True):
    def process_event(ht, w, thresh_min, thresh_max, pre_window, post_window, jump):
        if len(ht) == 0:
            return [], []

        triggered_hits_index = []
        triggered_hits_time = []

        ends   = ht + w
        right  = np.searchsorted(ht, ends, side="left")
        left   = np.arange(len(ht))
        counts = right - left

        # Posibles inicios de cluster
        trigger_indices = np.where((counts >= thresh_min) & (counts < thresh_max))[0]
        if len(trigger_indices) == 0:
            return [], []

        used_hits = set()
        last_trigger_time = -np.inf

        for idx in trigger_indices:
            if ht[idx] in used_hits:
                continue  # este hit ya pertenece a un cluster previo

            # Define los límites de ventana para este posible cluster
            t_min = ht[idx]
            t_max = ht[idx] + w

            indices_in_window = np.where((ht >= t_min) & (ht <= t_max))[0]
            if len(indices_in_window) < thresh_min:
                continue  # no alcanza el umbral inferior

            # Limita tamaño máximo de cluster
            if len(indices_in_window) > thresh_max:
                indices_in_window = indices_in_window[:thresh_max]

            hit_times_in_window = ht[indices_in_window]
            first_hit_time = hit_times_in_window[0]

            # Evita solapamientos temporales (dead time)
            if first_hit_time < last_trigger_time + jump:
                continue

            # Si algún hit de este cluster ya fue usado, se descarta
            if any(ht[i] in used_hits for i in indices_in_window):
                continue

            # Expande si hay pre/post window
            t_min_expanded = hit_times_in_window[0] - pre_window
            t_max_expanded = hit_times_in_window[-1] + post_window
            indices_final = np.where((ht >= t_min_expanded) & (ht <= t_max_expanded))[0]
            hit_times_final = ht[indices_final]

            # Guarda el cluster
            triggered_hits_index.append(indices_final)
            triggered_hits_time.append(hit_times_final)
            last_trigger_time = hit_times_final[-1]

            # Marca hits usados
            used_hits.update(ht[indices_final])

        return triggered_hits_index, triggered_hits_time

    # --- SINGLE EVENT ---
    if mode == "single_event":
        ht = ak.to_numpy(hit_times[event])
        idxs, times = process_event(ht, w, thresh_min, thresh_max, pre_window, post_window, jump)
        return {event: idxs}, {event: times}

    # --- MULTIPLE EVENTS ---
    elif mode == "multiple_events":
        nevents = len(hit_times)
        triggered_hits_index = {}
        triggered_hits_time  = {}

        for event in tqdm(range(nevents), total=nevents, leave=progress_bar):
            ht = ak.to_numpy(hit_times[event])
            if len(ht) == 0:
                triggered_hits_index[event] = []
                triggered_hits_time[event]  = []
                continue

            idxs, times = process_event(ht, w, thresh_min, thresh_max, pre_window, post_window, jump)
            triggered_hits_index[event] = idxs
            triggered_hits_time[event]  = times
            if len(idxs) > 0:
                triggered_hits_index[event] = idxs
                triggered_hits_time[event]  = times

        return triggered_hits_index, triggered_hits_time

    else:
        print("enter a valid mode name")
        return {}, {}

def candidatenHits(mode,
          hit_times,
          w,
          thresh_min,
          thresh_max,
          pre_window,
          post_window,
          jump,
          event=0,
          progress_bar=True):

    def process_event(ht,
                      w,
                      thresh_min,
                      thresh_max,
                      pre_window,
                      post_window,
                      jump):

        if len(ht) == 0:
            return [], [], [], []

        # Orden temporal
        ht = np.sort(ht)

        # ------------------------
        # Sliding window trigger
        # ------------------------

        ends = ht + w
        right = np.searchsorted(ht, ends, side="left")
        left = np.arange(len(ht))
        counts = right - left

        trigger_indices = np.where(counts >= thresh_min)[0]

        triggered_hits_index = []
        triggered_hits_time = []

        trigger_seeds = []      # t_seed (paper SLE crossing)
        roi_start_times = []   # primer hit físico del cluster

        last_cluster_end = -np.inf

        # ------------------------
        # Loop triggers
        # ------------------------

        for idx in trigger_indices:

            t_seed = ht[idx]

            # Deadtime basado en CLUSTER REAL
            if t_seed < last_cluster_end + jump:
                continue

            # ------------------------
            # ROI física
            # ------------------------

            t_min = t_seed - pre_window
            t_max = t_seed + w + post_window

            roi_mask = (ht >= t_min) & (ht <= t_max)
            indices_roi = np.where(roi_mask)[0]

            if len(indices_roi) < thresh_min:
                continue
            
            if thresh_max is not None and len(indices_roi) > thresh_max:
                continue

            hit_times_roi = ht[indices_roi]

            # ------------------------
            # Guardado
            # ------------------------

            triggered_hits_index.append(indices_roi)
            triggered_hits_time.append(hit_times_roi)

            trigger_seeds.append(t_seed)
            roi_start_times.append(hit_times_roi[0])

            # Deadtime real: fin del cluster físico
            last_cluster_end = hit_times_roi[-1]

        return triggered_hits_index, triggered_hits_time, trigger_seeds, roi_start_times

    # ==================================================
    # SINGLE EVENT
    # ==================================================

    if mode == "single_event":

        ht = ak.to_numpy(hit_times[event])

        idxs, times, seeds, roi_starts = process_event(
            ht,
            w,
            thresh_min,
            thresh_max,
            pre_window,
            post_window,
            jump
        )

        return ({event: idxs},
                {event: times},
                {event: seeds},
                {event: roi_starts})

    # ==================================================
    # MULTIPLE EVENTS
    # ==================================================

    elif mode == "multiple_events":

        nevents = len(hit_times)

        triggered_hits_index = {}
        triggered_hits_time = {}
        trigger_seeds_dict = {}
        roi_start_times_dict = {}

        for event in tqdm(range(nevents),
                          total=nevents,
                          leave=progress_bar):

            ht = ak.to_numpy(hit_times[event])

            if len(ht) == 0:
                triggered_hits_index[event] = []
                triggered_hits_time[event] = []
                trigger_seeds_dict[event] = []
                roi_start_times_dict[event] = []
                continue

            idxs, times, seeds, roi_starts = process_event(
                ht,
                w,
                thresh_min,
                thresh_max,
                pre_window,
                post_window,
                jump
            )

            triggered_hits_index[event] = idxs
            triggered_hits_time[event] = times
            trigger_seeds_dict[event] = seeds
            roi_start_times_dict[event] = roi_starts

        return (triggered_hits_index,
                triggered_hits_time,
                trigger_seeds_dict,
                roi_start_times_dict)

    else:
        print("enter a valid mode name")
        return {}, {}, {}, {}

def select_signal_clusters(triggered_hit_times, triggered_hit_indices, trms_cut_inf, trms_cut_sup, noHits_cut_inf, noHits_cut_sup):
    signal_clusters = []
    signal_indices  = []
    len_signal_clusters = []

    for k,v in tqdm(triggered_hit_times.items(), total=len(triggered_hit_times.items()), leave=True):
            event_signal_clusters = []
            event_signal_indices  = []
            event_len_signal_clusters = []
            for i,j in zip(triggered_hit_times.get(k), triggered_hit_indices.get(k)):
                if np.std(i, ddof=0) > trms_cut_inf and np.std(i, ddof=0) < trms_cut_sup and noHits_cut_inf < len(i) <= noHits_cut_sup:
                    event_signal_indices.append(j)
                    event_signal_clusters.append(i)
                    event_len_signal_clusters.append(len(i))

            signal_clusters.append(event_signal_clusters)
            signal_indices.append(event_signal_indices)
            len_signal_clusters.append(event_len_signal_clusters)

    signal_indices = ak.Array([ak.ravel(signal_indices[i]) for i in range(len(signal_indices))])
    len_signal_clusters = ak.Array(len_signal_clusters)

    return signal_indices, len_signal_clusters

def filter_charge(events, charges, data, min_charge, max_charge):
    # 1. Compute the summed charge per event_id
    event_ids, sums = np.unique(events, return_inverse=False, return_counts=False), np.zeros(len(np.unique(events)))
    sums = np.bincount(events.astype(int), weights=charges)

    # 2. Create the mask (per event)
    mask = (sums > min_charge) & (sums < max_charge)
    selected_event_ids = np.where(mask)[0]

    # 3. Create the whole mask
    hit_mask = np.isin(events, selected_event_ids)

    # 4. Apply Mask
    filtered_data = data[hit_mask]

    return filtered_data