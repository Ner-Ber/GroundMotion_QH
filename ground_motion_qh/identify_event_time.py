import pandas as pd
import torch
import seisbench
import seisbench.models as sbm
import pandas as pd
from datetime import timedelta
from obspy import UTCDateTime


def identify_ps_event_time(
    st, #stream
    pmin=0.5,  # mininum of the probability of p-phase to identify events
    smin=0.5, # mininum of the probability of s-phase to identify events
    max_pstime=7 #Threshold of maximum S-P time 
):
    #define models
    picker = sbm.PickBlue("phasenet")

    if torch.cuda.is_available():
        picker.cuda()


    picks = picker.classify(st, batch_size=256, P_threshold=pmin, S_threshold=smin).picks

    # convert the picks and station metadata into pandas dataframes
    pick_df = []
    for i,p in enumerate(picks):
        pick_df.append({
            "pick_idx":i, #index of the phases
            "id": p.trace_id, #station name
            "timestamp": p.peak_time.datetime, # time of the phases
            "amp":0.0, # dummy
            "prob": p.peak_value, # probability of the phases
            "type": p.phase.lower() # p or s phases
        })
    pick_df = pd.DataFrame(pick_df)

    #Exclude bad events
    #Only use events which have s and p phases

    hyp_df=[]
    pflg=0
    ev_id=0
    for index,row in pick_df.iterrows():
        if row['type']=='p':
            ptime=row['timestamp']
            pflg=1
        if pflg==1 and row['type']=='s' and row['timestamp']-timedelta(seconds=max_pstime)<ptime:
            hyp_df.append({
                    "ev_id":ev_id, # index of the events
                    "ptime":ptime, # time of the phases
                    'pstime':UTCDateTime(row['timestamp'])-UTCDateTime(ptime) # s-p time, propotional to distance
                })
            pflg=0
            ev_id+=1
        
    hyp_df = pd.DataFrame(hyp_df)   

    return pick_df,hyp_df
