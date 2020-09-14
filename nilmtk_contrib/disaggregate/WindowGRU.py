from __future__ import print_function, division
from warnings import warn, filterwarnings

from matplotlib import rcParams
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import sys
import pandas as pd
import numpy as np
import h5py
import os
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

import random
random.seed(10)
np.random.seed(10)
class WindowGRU(Disaggregator):

    def __init__(self, params):

        self.MODEL_NAME = "WindowGRU"
        self.save_model_path = params.get('save-model-path',None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.max_val = 800
        self.batch_size = params.get('batch_size',512)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')

        train_main = pd.concat(train_main,axis=0).values
        train_main = train_main.reshape((-1, self.sequence_length, 1))
        new_train_appliances  = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0).values
            app_df = app_df.reshape((-1, 1))
            new_train_appliances.append((app_name, app_df))

        train_appliances = new_train_appliances
        for app_name, app_df in train_appliances:
            if app_name not in self.models:
                print("First model training for", app_name)
                self.models[app_name] = self.return_network()
            else:
                print("Started re-training model for", app_name)

            model = self.models[app_name]
            mains = train_main.reshape((-1,self.sequence_length,1))
            app_reading = app_df.reshape((-1,1))
            filepath = 'windowgru-temp-weights-'+str(random.randint(0,100000))+'.h5'
            checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
            model.fit(
                    mains, app_reading,
                    validation_split=.15,
                    epochs=self.n_epochs,
                    batch_size=self.batch_size,
                    callbacks=[ checkpoint ],
                    shuffle=True,
            )
            model.load_weights(filepath)

    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):

        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')
        
        test_predictions = []
        for mains in test_main_list:
            disggregation_dict = {}
            mains = mains.values.reshape((-1,self.sequence_length,1))
            for appliance in self.models:
                prediction = self.models[appliance].predict(mains,batch_size=self.batch_size)
                prediction = np.reshape(prediction, len(prediction))
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                valid_predictions = self._denormalize(valid_predictions, self.max_val)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        max_val = self.max_val
        if method == 'train':
            print("Training processing")
            processed_mains = []

            for mains in mains_lst:
                # add padding values
                padding = [0 for i in range(0, self.sequence_length - 1)]
                paddf = pd.DataFrame({mains.columns.values[0]: padding})
                mains = mains.append(paddf)
                mainsarray = self.preprocess_train_mains(mains)
                processed_mains.append(pd.DataFrame(mainsarray))

            tuples_of_appliances = []
            for (appliance_name, app_dfs_list) in submeters_lst:
                processed_app_dfs = []
                for app_df in app_dfs_list:                    
                    data = self.preprocess_train_appliances(app_df)
                    processed_app_dfs.append(pd.DataFrame(data))
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains , tuples_of_appliances

        if method == 'test':
            processed_mains = []
            for mains in mains_lst:                
                # add padding values
                padding = [0 for i in range(0, self.sequence_length - 1)]
                paddf = pd.DataFrame({mains.columns.values[0]: padding})
                mains = mains.append(paddf)
                mainsarray = self.preprocess_test_mains(mains)
                processed_mains.append(pd.DataFrame(mainsarray))

            return processed_mains

    def preprocess_test_mains(self, mains):

        mains = self._normalize(mains, self.max_val)
        mainsarray = np.array(mains)
        indexer = np.arange(self.sequence_length)[
            None, :] + np.arange(len(mainsarray) - self.sequence_length + 1)[:, None]
        mainsarray = mainsarray[indexer]
        mainsarray = mainsarray.reshape((-1,self.sequence_length))
        return pd.DataFrame(mainsarray)

    def preprocess_train_appliances(self, appliance):

        appliance = self._normalize(appliance, self.max_val)
        appliancearray = np.array(appliance)
        appliancearray = appliancearray.reshape((-1,1))
        return pd.DataFrame(appliancearray)

    def preprocess_train_mains(self, mains):

        mains = self._normalize(mains, self.max_val)
        mainsarray = np.array(mains)
        indexer = np.arange(self.sequence_length)[None, :] + np.arange(len(mainsarray) - self.sequence_length + 1)[:, None]
        mainsarray = mainsarray[indexer]
        mainsarray = mainsarray.reshape((-1,self.sequence_length))
        return pd.DataFrame(mainsarray)

    def _normalize(self, chunk, mmax):

        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):

        tchunk = chunk * mmax
        return tchunk

    def return_network(self):
        '''Creates the GRU architecture described in the paper
        '''
        model = Sequential()
        # 1D Conv
        model.add(Conv1D(16,4,activation='relu',input_shape=(self.sequence_length,1),padding="same",strides=1))
        # Bi-directional GRUs
        model.add(Bidirectional(GRU(64, activation='relu',
                                    return_sequences=True), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Bidirectional(GRU(128, activation='relu',
                                    return_sequences=False), merge_mode='concat'))
        model.add(Dropout(0.5))
        # Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model
