import numpy as np
import pandas as pd

class ApplianceNotFoundError(Exception):
    """Custom exception for when appliance parameters are not found."""
    pass

def preprocess(sequence_length=None, mains_mean=None, mains_std=None, mains_lst=None, submeters_lst=None, method="train", appliance_params=None, windowing=False):
    """
    Preprocesses mains and appliance data by creating sliding windows and normalizing the data.

    Args:
        sequence_length (int): The length of the sliding window.
        mains_mean (float): The mean of the mains data for normalization.
        mains_std (float): The standard deviation of the mains data for normalization.
        mains_lst (list of pd.DataFrame): A list of DataFrames, each containing mains data.
        submeters_lst (list of tuples): A list where each tuple contains the appliance name 
                                        (str) and a list of its corresponding DataFrames.
        method (str, optional): The mode of operation, either "train" or "test". Defaults to "train".
        appliance_params (dict, optional): A dictionary containing the mean and std for each 
                                           appliance. Required if method is "train". Defaults to None.
        windowing (bool, optional): If True, applies sliding window to appliance data. 
                                    If False, normalizes the flattened appliance data. Defaults to False.

    Returns:
        If method is "test" or submeters_lst is not provided:
            list of pd.DataFrame: A list of preprocessed mains dataframes.
        If method is "train":
            tuple: A tuple containing:
                - list of pd.DataFrame: Preprocessed mains data.
                - list of tuples: Preprocessed appliance data, structured like submeters_lst.
    """
    pad = sequence_length // 2

    # Preprocess mains data
    proc_mains = []
    for mains in mains_lst:
        v = mains.values.flatten()
        # Pad the sequence to handle windowing at the edges
        v = np.pad(v, (pad, pad), 'constant', constant_values=(0,0))
        # Create sliding windows
        windows = np.array([v[i:i+sequence_length] for i in range(len(v) - sequence_length + 1)], dtype=np.float32)
        # Normalize the windows
        windows = (windows - mains_mean) / mains_std
        proc_mains.append(pd.DataFrame(windows))

    # Return only mains data if in test mode or no appliance data is provided
    if method == "test" or not submeters_lst:
        return proc_mains
    
    # Preprocess appliance data
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
                # Apply padding and sliding window if specified
                flat = np.pad(flat, (pad, pad), 'constant', constant_values=(0,0))
                windows = np.array([flat[i:i+sequence_length] for i in range(len(flat) - sequence_length + 1)], dtype=np.float32)
                windows = (windows - mean) / std
                sub.append(pd.DataFrame(windows))
            else:
                # Normalize the flattened data directly
                flat = (flat - mean) / std
                sub.append(pd.DataFrame(flat.reshape(-1, 1)))
        proc_apps.append((app_name, sub))
    
    return proc_mains, proc_apps