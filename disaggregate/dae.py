from __future__ import print_function, division
from warnings import warn

from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict 

from keras.optimizers import SGD
from keras.models import Sequential
import matplotlib.pyplot as  plt


class DAE(Disaggregator):
    
    def __init__(self, sequence_length=300, learning_rate=1e-3, n_epochs = 100):
        
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.n_epochs = n_epochs
        self.trained = False
        self.models = OrderedDict()
        self.std_of_random_sample = None
        self.appliance_wise_max = OrderedDict()
        self.default_max_reading = 10000
        
    def partial_fit(self, train_main, train_appliances, do_preprocessing = True, chunk_id=None, **load_kwargs):


        train_main = train_main.fillna(0)

        print (train_main.shape)

        new_train_appliances = []

        for index, (appliance_name, power) in enumerate(train_appliances):

            print (power.shape)

            new_train_appliances.append((appliance_name,power.fillna(0)))

        train_appliances = new_train_appliances + []

        
        print("...............DAE partial_fit running...............")

        if do_preprocessing:
            
            #print (type(train_main))
            
            #print (train_main.shape)
            
            
            train_main = self.preprocess_input(self.choose_windows(train_main.values))
            
            #print (train_main.shape)
            
            new_train_appliances = []
            
            for index, (appliance_name, power) in enumerate(train_appliances):
                
                # Modifying the power
                #print (power.shape)
                
                preprocessed_power = self.preprocess_output(self.choose_windows(power.values), appliance_name)
                    
                new_train_appliances.append((appliance_name,preprocessed_power))
            
            train_appliances = new_train_appliances
            
        # If the data is already preprocessed
        
        for appliance_name, power in train_appliances:
            
            if appliance_name not in self.models:
                print ("New training for ",appliance_name)
                self.models[appliance_name] = self.return_network()
            
            else:
                
                print ("Retrained model for ",appliance_name)
                
            model = self.models[appliance_name]
            #print (train_main.shape)
            #print (power.shape)
            
            print (np.max(train_main))
            print (np.max(power))
            
            model.fit(train_main, power, epochs = self.n_epochs)

            
            

    def disaggregate_chunk(self, test_main, do_preprocessing=True):
        
        original_test_main = test_main.copy()
        
        if do_preprocessing:
            
            test_main = self.preprocess_input(self.choose_windows(test_main.values))
        
        disaggregated_power_dict = {}
        
        for appliance in self.models:
            
            prediction = self.models[appliance].predict(test_main)
            
            prediction = prediction * self.appliance_wise_max[appliance]
            
            valid_predictions = prediction.flatten()[:original_test_main.size]
            
            valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
            
            df = pd.Series(valid_predictions,index=original_test_main.index)

            disaggregated_power_dict[appliance] = df
            #tuples_of_predictions.append((appliance, df))

        
        return pd.DataFrame(disaggregated_power_dict,dtype='float32')
        

            
 
    def choose_windows(self, data):

        excess_entries =  self.sequence_length - (data.size % self.sequence_length)
        
        lst = np.array([0] * excess_entries)
        # print("lst...")
        # print(lst)
        # print(len(lst))
        # print (data.shape)
        arr = np.concatenate((data.flatten(), lst),axis=0)
        # print("arr...")
        # print(arr)
        # print(len(arr))
        return arr.reshape((-1,self.sequence_length))

        
    def preprocess_input(self,windowed_x):
        
        mean_sequence = np.mean(windowed_x,axis=1).reshape((-1,1))
        
        windowed_x = windowed_x - mean_sequence # Mean centering each sequence
        
#         if self.std_of_random_sample==None:

#             std_of_random_sample = 0

#             while std_of_random_sample==0:

#                 random_index = np.random.randint(0,len(windowed_x))
                
#                 std_of_random_sample = np.std(windowed_x[random_index])
            
            
#             self.std_of_random_sample = std_of_random_sample
        
        
        self.std_of_random_sample = self.default_max_reading

        print ( self.std_of_random_sample)
        
        return (windowed_x/self.std_of_random_sample).reshape((-1,self.sequence_length,1))
    
        
        
    def preprocess_output(self, windowed_y, appliance_name):
        
        if appliance_name not in self.appliance_wise_max:
            
            self.appliance_wise_max[appliance_name] = self.default_max_reading
        
        return (windowed_y/self.appliance_wise_max[appliance_name]).reshape((-1,self.sequence_length,1))
        
    def return_network(self):
        
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(8, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(Flatten())

        # Fully Connected Layers
        model.add(Dropout(0.2))
        model.add(Dense((self.sequence_length)*8, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense((self.sequence_length)*8, activation='relu'))

        model.add(Dropout(0.2))

        # 1D Conv
        model.add(Reshape(((self.sequence_length), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

        optimizer = SGD(lr=self.learning_rate)
        
        model.compile(loss='mse', optimizer='adam')

        return model
