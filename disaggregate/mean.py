from __future__ import print_function, division
from warnings import warn

import pandas as pd
import numpy as np

from nilmtk.disaggregate import Disaggregator


class Mean(Disaggregator):
    """1 dimensional baseline Mean algorithm.

    Attributes
    ----------
    model : list of dicts
       Each dict has these keys:
           mean : list of mean values, one for each appliance (the mean power
           (Watts)) training_metadata : The appliance type (and perhaps some
           other metadata) for each model.

    MIN_CHUNK_LENGTH : int

    MODEL_NAME = string
    """

    def __init__(self, d):
        self.model = []
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = 'Mean'

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        '''
            train_main :- pd.DataFrame It will contain the mains reading.
            train_appliances :- list of tuples [('appliance1',df1),('appliance2',df2),...]

        '''
        train_main = pd.concat(train_main, axis=0)
        train_app_tmp = []

        for app_name, df_list in train_appliances:
            df_list = pd.concat(df_list, axis=0)
            train_app_tmp.append((app_name,df_list))

        train_appliances = train_app_tmp

        print("...............Mean partial_fit running...............")

        for appliance_name, power in train_appliances:

            # there will be only mean state for all appliances. 
            # the algorithm will always predict mean power

            mean = np.nanmean(power)
            mean = np.round(mean).astype(np.int32)

            self.model.append({
                    'mean': mean,
                    'training_metadata': appliance_name})

    def disaggregate_chunk(self, test_mains):

        print("...............Mean disaggregate_chunk running...............")

        test_predictions_list = []

        for test_df in test_mains:

            if len(test_mains) < self.MIN_CHUNK_LENGTH:
                raise RuntimeError("Chunk is too short.")

            appliance_powers_dict = {}
            for i, model in enumerate(self.model):
                print("Estimating power demand for '{}'"
                      .format(model['training_metadata']))
                # a list of predicted power values for ith appliance            
                predicted_power = [self.model[i]['mean'] for j in range(0, test_mains.shape[0])]
                column = pd.Series(predicted_power, index=test_mains.index, name=i)
                appliance_powers_dict[self.model[i]['training_metadata']] = column

            appliance_powers = pd.DataFrame(appliance_powers_dict, dtype='float32')

            test_predictions_list.append(appliance_powers)


        return test_predictions_list
