from __future__ import print_function, division
from warnings import warn

import pandas as pd
import numpy as np
import pickle
import copy

from nilmtk.utils import find_nearest
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

SEED = 42
np.random.seed(SEED)


class Mean(Disaggregator):
    

    def __init__(self):
        self.model = []
        self.state_combinations = None
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = 'Mean'



    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        
        print("...............Function in partial_fit mean  .............")
        
        num_on_states=None
        if len(train_appliances)>12:
            max_num_clusters=2
        else:
            max_num_clusters=3
        appliance_in_model=[d['training_metadata'] for d in self.model]
        for appliance,readings in train_appliances:
            if appliance in appliance_in_model:
                raise RuntimeError(
                "Appliance {} is already in model!"
                "  Can't train twice on the same meter!",appliance)

            
            states=readings.mean(skipna=True)
            states = np.round(states).astype(np.int32)
            self.model.append({
                    'states': states,
                    'training_metadata': appliance})
            


    def _set_state_combinations_if_necessary(self):
        """Get centroids"""
        # If we import sklearn at the top of the file then auto doc fails.
        if (self.state_combinations is None or
                self.state_combinations.shape[1] != len(self.model)):
            from sklearn.utils.extmath import cartesian
            centroids = [model['states'] for model in self.model]
            self.state_combinations = cartesian(centroids)

    def disaggregate_chunk(self, mains):

        if len(mains) < self.MIN_CHUNK_LENGTH:
            raise RuntimeError("Chunk is too short.")

        
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        
        self._set_state_combinations_if_necessary()



        state_combinations = self.state_combinations
        summed_power_of_each_combination = np.sum(state_combinations, axis=1)


        # Start disaggregation
        indices_of_state_combinations, residual_power = find_nearest(
            summed_power_of_each_combination, mains.values)

        appliance_powers_dict = {}
        for i, model in enumerate(self.model):
            print("Estimating power demand for '{}'"
                  .format(model['training_metadata']))
            predicted_power = state_combinations[
                indices_of_state_combinations, i].flatten()
            column = pd.Series(predicted_power, index=mains.index, name=i)
            appliance_powers_dict[self.model[i]['training_metadata']] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict, dtype='float32')
        return appliance_powers

