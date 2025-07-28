from nilmtk import DataSet
import numpy as np
import pandas as pd

def calculate_multi_building_mains_stats(dataset_path, building_ids, start_time, end_time, 
                                        ac_type='active', sample_period=60):
    """
    Calculates mains statistics across multiple buildings by combining their data.
    """
    ds = DataSet(dataset_path)
    ds.set_window(start=start_time, end=end_time)

    all_mains_data = []

    # 1. Loop through each specified building ID
    for building_id in building_ids:
        print(f"Processing Building {building_id}...")
        try:
            mains = ds.buildings[building_id].elec.mains()
            
            # Use power_series_all_data for simplicity, it handles the generator loop internally
            power_data = mains.power_series_all_data(
                ac_type=ac_type,
                sample_period=sample_period
            )

            if power_data is not None and not power_data.empty:
                all_mains_data.append(power_data)
            else:
                print(f"  - No data found for Building {building_id} in the specified timeframe.")

        except KeyError:
            print(f"  - Building {building_id} not found in the dataset.")
        except Exception as e:
            print(f"  - An error occurred for Building {building_id}: {e}")

    # 2. Check if any data was collected
    if not all_mains_data:
        print("Could not retrieve data for any of the specified buildings.")
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'data_points': 0}

    # 3. Concatenate all data into a single pandas Series
    print("\nCombining data from all buildings...")
    combined_data = pd.concat(all_mains_data)
    clean_data = combined_data.dropna()

    # 4. Calculate statistics on the combined data
    stats = {
        'mean': clean_data.mean(),
        'std': clean_data.std(),
        'min': clean_data.min(),
        'max': clean_data.max(),
        'data_points': len(clean_data),
        'ac_type': ac_type
    }
    
    ds.store.close()
    return stats

stats = calculate_multi_building_mains_stats(
    dataset_path="/home/ubuntu/downloads/refit.h5",
    building_ids=[2],  # Pass a list of buildings
    start_time='2014-04-01',
    end_time='2014-04-30',
    ac_type='active',      # Pass 'active' as a string
    sample_period=60
)

print("\n--- Combined Mains Statistics ---")
if stats['data_points'] > 0:
    print(f"Combined Mains Mean: {stats['mean']:.2f}W")
    print(f"Combined Mains Std: {stats['std']:.2f}W")
    print(f"Data Range: {stats['min']:.2f}W to {stats['max']:.2f}W")
    print(f"Total Data Points from all buildings: {stats['data_points']}")
else:
    print("No data available to calculate statistics.")