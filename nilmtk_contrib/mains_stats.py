from nilmtk import DataSet
import numpy as np
import pandas as pd


def calculate_exact_mains_stats_v2(dataset_path, building_id, start_time, end_time, 
                                   ac_type='active', sample_period=60):
    """
    Calculate mains statistics using correct NILMTK API
    """
    ds = DataSet(dataset_path)
    building = ds.buildings[building_id]
    mains = building.elec.mains()
    
    # Check available AC types
    available_types = mains.available_ac_types('power')
    print(f"Available AC types: {available_types}")
    
    if ac_type not in available_types:
        print(f"Warning: {ac_type} not available. Using {available_types[0]}")
        ac_type = available_types[0]
    
    print(f"Using AC type: {ac_type}")
    
    # Set the timeframe window for the dataset
    ds.set_window(start=start_time, end=end_time)
    
    # Use power_series method which is the correct approach
    try:
        power_data_generator = mains.power_series(
            ac_type=ac_type,
            sample_period=sample_period
        )
        
        # Collect all power data from the generator
        power_data_list = []
        for chunk in power_data_generator:
            power_data_list.append(chunk)
        
        # Concatenate all chunks
        if power_data_list:
            power_data = pd.concat(power_data_list)
        else:
            power_data = pd.Series(dtype=float)
            
    except Exception as e:
        print(f"Error with power_series method: {e}")
        print("Trying alternative approach with load...")
        
        # Alternative: Use load method
        try:
            data_generator = mains.load(
                physical_quantity='power',
                ac_type=ac_type,
                sample_period=sample_period
            )
            
            # Collect all data from the generator
            data_list = []
            for chunk in data_generator:
                data_list.append(chunk)
            
            if data_list:
                df = pd.concat(data_list)
                # Get the power column
                power_column = ('power', ac_type)
                power_data = df[power_column]
            else:
                power_data = pd.Series(dtype=float)
                
        except Exception as e2:
            print(f"Error with load method: {e2}")
            power_data = pd.Series(dtype=float)
    
    # Clean data
    clean_data = power_data.dropna()
    
    # Calculate statistics
    mains_mean = clean_data.mean()
    mains_std = clean_data.std()
    mains_min = clean_data.min()
    mains_max = clean_data.max()
    data_points = len(clean_data)
    
    return {
        'mean': mains_mean,
        'std': mains_std,
        'min': mains_min,
        'max': mains_max,
        'data_points': data_points,
        'ac_type': ac_type
    }


stats = calculate_exact_mains_stats_v2(
    dataset_path= r"/home/ubuntu/downloads/ukdale.h5",
    building_id=1,
    start_time='2013-04-01',
    end_time='2013-04-30',
    ac_type=['active'],
    sample_period=60
)

print(f"Exact Mains Mean: {stats['mean']:.2f}W")
print(f"Exact Mains Std: {stats['std']:.2f}W")
print(f"Data Range: {stats['min']:.2f}W to {stats['max']:.2f}W")
print(f"Total Data Points: {stats['data_points']}")
print(f"AC Type Used: {stats['ac_type']}")