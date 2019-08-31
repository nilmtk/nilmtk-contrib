from __future__ import print_function, division
from warnings import warn
import pandas as pd
import numpy as np
import json
from nilmtk.disaggregate import Disaggregator
import os
class Mean(Disaggregator): # Add the class name for the algorithm
    def __init__(self, model_parameters):
        # model_parameters is the set of parameters used to initialize mean algorithm
        self.model = {}
        self.MODEL_NAME = 'Mean' # Add the name for the algorithm
        self.save_model_path = model_parameters.get('save-model-path',None)
        self.load_model_path = model_parameters.get('pretrained-model-path',None)
        self.chunk_wise_training = model_parameters.get('chunk_wise_training',True)
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        # train_main = pd.concat(train_main,axis=0) # This is for getting the mains data
        for appliance_name, power in train_appliances:
            power_ = pd.concat(power,axis=0) # This is the power column corresponding to appliance_name
            appliance_dict = self.model.get(appliance_name,{'sum':0,'n_elem':0})
            appliance_dict['sum']+=int(np.nansum(power_.values))
            appliance_dict['n_elem']+=len(power_[~np.isnan(power_)])
            self.model[appliance_name] = appliance_dict 
        if self.save_model_path:
            self.save_model(self.save_model_path)

    def disaggregate_chunk(self, test_mains):
        test_predictions_list = []
        for test_df in test_mains:

            appliance_powers = pd.DataFrame()
            """
            Use the test_df and obtain predictions. 
            Store the results as pandas dataframe with columns as [appliance_1,appliance_2,...]
            Add this df to test_predictions_list
            """
            for i, appliance_name in enumerate(self.model):
                # Modify the algorithm from here onwards
                # Obtain the prediction for the test_df, which is a mains dataframe           
                model = self.model[appliance_name]
                predicted_power = [model['sum']/model['n_elem'] for j in range(0, test_df.shape[0])]
                # Store the prediction for the appliance 
                appliance_powers[appliance_name] = pd.Series(predicted_power, index=test_df.index, name=i)
            test_predictions_list.append(appliance_powers)
        return test_predictions_list

    def save_model(self,folder_name):
        string_to_save = json.dumps(self.model)
        os.makedirs(folder_name,exist_ok=True)
        with open(os.path.join("folder_name","model.txt","w")) as f:
            f.write(string_to_save)

    def load_model(self,folder_name):
        file_name = folder_name+"/model.txt"
        with open(folder_name+"/model.txt","r") as f:
            model_string = f.read().strip()
            self.model = json.loads(model_string)
