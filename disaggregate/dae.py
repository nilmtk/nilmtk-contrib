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
import keras.backend as K
from statistics import mean
import os
import pickle
import random
class DAE(Disaggregator):
    
    def __init__(self, params):
        self.MODEL_NAME = "DAE"
        self.save_model_path = params.get('save-model-path',None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1000
        self.mains_std = 1800
        self.batch_size = 512
        self.appliance_params = params.get('appliance_params',{})

        
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True,**load_kwargs):
        
        print("...............DAE partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        if do_preprocessing:
            print ("Doing Preprocessing")
            train_main,train_appliances = self.call_preprocessing(train_main,train_appliances,'train')
        train_main = pd.concat(train_main,axis=0).values
        train_main = train_main.reshape((-1,self.sequence_length,1))#np.array([i.values.reshape((self.sequence_length,1)) for i in train_main])
        new_train_appliances  = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df,axis=0).values
            app_df = app_df.reshape((-1,self.sequence_length,1))#np.array([i.values.reshape((self.sequence_length,1)) for i in app_df])
            new_train_appliances.append((app_name, app_df))
        train_appliances = new_train_appliances
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print ("First model training for ",appliance_name)
                self.models[appliance_name] = self.return_network()
                print (self.models[appliance_name].summary())
            print ("Started Retraining model for ",appliance_name)    
            model = self.models[appliance_name]
            filepath = 'dae-temp-weights-'+str(random.randint(0,10))+'.h5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            train_x,v_x,train_y,v_y = train_test_split(train_main,power,test_size=.15)
            model.fit(train_x,train_y,validation_data = [v_x,v_y],epochs = self.n_epochs, callbacks = [checkpoint],shuffle=True,batch_size=self.batch_size)
            model.load_weights(filepath)
            if not os.path.exists('dae'):
                    os.mkdir('dae')
            pickle_out = open("dae/" + appliance_name + ".pickle", "wb")
            pickle.dump(model, pickle_out)
            pickle_out.close()

    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list,submeters=None,method='test')
        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1,self.sequence_length,1))
            disggregation_dict = {}
            for appliance in self.models:
                prediction = self.models[appliance].predict(test_main)
                app_mean = self.appliance_params[appliance]['mean']
                app_std = self.appliance_params[appliance]['std']
                prediction = self.denormalize_output(prediction,app_mean,app_std)
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
                series = pd.Series(valid_predictions)
                disggregation_dict[appliance] = series
            results = pd.DataFrame(disggregation_dict,dtype='float32')
            test_predictions.append(results)
        return test_predictions
            

    def return_network(self):
        
        model = Sequential()
        model.add(Conv1D(8, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(Flatten())
        model.add(Dense((self.sequence_length)*8, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense((self.sequence_length)*8, activation='relu'))
        model.add(Reshape(((self.sequence_length), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))
        model.compile(loss='mse', optimizer='adam')
        return model

    def call_preprocessing(self, mains, submeters, method):
        sequence_length  = self.sequence_length
        if method=='train':
            mains = pd.concat(mains,axis=0)
            mains = self.normalize_input(mains.values,sequence_length,self.mains_mean,self.mains_std,True)
            mains_df_list = [pd.DataFrame(mains)]
            tuples_of_appliances = []
            for (appliance_name,df) in submeters:
                df = pd.concat(df,axis=0)
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                
                data = self.normalize_output(df.values, sequence_length,app_mean,app_std,True)
                appliance_df_list  = [pd.DataFrame(data)]
                tuples_of_appliances.append((appliance_name, appliance_df_list))
            return mains_df_list, tuples_of_appliances

        if method=='test':

            mains = pd.concat(mains,axis=0)
            mains = self.normalize_input(mains.values ,sequence_length,self.mains_mean,self.mains_std,False)
            mains_df_list = [pd.DataFrame(mains)]
            return mains_df_list
    
        
    def normalize_input(self,data,sequence_length, mean, std, overlapping=False):
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0)   
        if overlapping:
            windowed_x = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_x = arr.reshape((-1,sequence_length))
        windowed_x = windowed_x - mean
        return (windowed_x/std).reshape((-1,sequence_length))

    def normalize_output(self,data,sequence_length, mean, std, overlapping=False):
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0) 
        if overlapping:  
            windowed_y = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_y = arr.reshape((-1,sequence_length))        
        windowed_y = windowed_y - mean
        return (windowed_y/std).reshape((-1,sequence_length))

    def denormalize_output(self,data,mean,std):
        return mean + data*std
    
    def set_appliance_params(self,train_appliances):

        for (app_name,df_list) in train_appliances:
            l = np.array(df_list[0])
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})