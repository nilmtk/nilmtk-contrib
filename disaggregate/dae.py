from __future__ import print_function, division
from warnings import warn

from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten

import pandas as pd
import numpy as np
from collections import OrderedDict 

from keras.optimizers import SGD
from keras.models import Sequential
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


class DAE(Disaggregator):
    
    def __init__(self, d):

        self.learning_rate = 1e-3
        self.sequence_length = 300
        self.n_epochs = 100
        self.trained = False
        self.models = OrderedDict()
        self.max_value = 6000

        if 'learning_rate' in d: 
        	self.learning_rate = d['learning_rate']
        if 'sequence_length' in d: 
        	self.sequence_length = d['sequence_length']
        if 'n_epochs' in d: 
        	self.n_epochs = d['n_epochs']
        if 'max_val' in d:
            self.max_val = d['max_val']
        
        
    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        
        print("...............DAE partial_fit running...............")

        train_main = train_main.values.reshape((-1,self.sequence_length,1))
        
        new_train_appliances  = []
        
        for app_name, app_df in train_appliances:
            
            new_train_appliances.append((app_name, app_df.values.reshape((-1,self.sequence_length,1))))
            
        train_appliances = new_train_appliances
        
        for appliance_name, power in train_appliances:
            
            if appliance_name not in self.models:
                print ("First model training for ",appliance_name)
                self.models[appliance_name] = self.return_network()
            
            else:

                print ("Started Retraining model for ",appliance_name)
                
            model = self.models[appliance_name]
            #print (train_main.shape)
            #print (power.shape)
            
            
            #print (np.max(train_main),np.max(power),np.min(train_main),np.min(power),np.isnan(train_main).any(),np.isnan(power).any())
            
            if len(train_main>10):
                # Do validation when you have sufficient samples

                filepath = 'temp-weights.h5'
                
                checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                train_x,v_x,train_y,v_y = train_test_split(train_main,power,test_size=.15)
                model.fit(train_x,train_y,validation_data = [v_x,v_y],epochs = self.n_epochs, callbacks = [checkpoint])

                model.load_weights(filepath)
            else:

                model.fit(train_main, power, epochs = self.n_epochs)
            
            
   

    def disaggregate_chunk(self, test_main):
        
        
        test_main = test_main.values
        
        test_main = test_main.reshape((-1,self.sequence_length,1))
        
        disggregation_dict = {}
        
        for appliance in self.models:
            
            prediction = self.models[appliance].predict(test_main)
            
            prediction = prediction * self.max_value
            
            valid_predictions = prediction.flatten()
            
            valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
            
            df = pd.Series(valid_predictions)

            disggregation_dict[appliance] = df
            
        return pd.DataFrame(disggregation_dict,dtype='float32')
        

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