import numpy as np
import pandas as pd

class ApplianceNotFoundError(Exception):
    pass

def preprocess(sequence_length = None,mains_mean = None,mains_std = None,mains_lst = None,submeters_lst = None,method="train",appliance_params=None,windowing=False):
    pad = sequence_length // 2

    proc_mains = []

    for mains in mains_lst:
        v = mains.values.flatten()
        v = np.pad(v,(pad,pad))
        windows = np.array([v[i:i+sequence_length] for i in range(len(v)-sequence_length + 1)],dtype=np.float32)
        windows = (windows - mains_mean)/mains_std
        proc_mains.append(pd.DataFrame(windows))
    if method == "test" or not submeters_lst:
        return proc_mains
    
    proc_apps = []

    for app_name, df_list in submeters_lst:
        if appliance_params is None or app_name not in appliance_params:
            raise ApplianceNotFoundError(f"Parameters for {app_name} not initialized.")

        mean = appliance_params[app_name]["mean"]
        std = appliance_params[app_name]["std"]

        sub = []

        for df in df_list:
            flat = df.values.flatten()


            if windowing:
                flat = np.pad(flat,(pad,pad))
                windows = np.array([flat[i:i+sequence_length] for i in range(len(flat)-sequence_length+1)],dtype=np.float32)
                windows = (windows-mean)/std
                sub.append(pd.DataFrame(windows))
            else:
                flat = (flat-mean)/std
                sub.append(pd.DataFrame(flat.reshape(-1,1)))
        proc_apps.append((app_name,sub))
    
    return proc_mains, proc_apps