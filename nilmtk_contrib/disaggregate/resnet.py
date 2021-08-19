from __future__ import print_function, division
from warnings import warn

from keras.layers.convolutional import Conv2D, ZeroPadding1D,MaxPooling1D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling1D

from nilmtk.disaggregate import Disaggregator
from keras.layers import Layer,Conv1D, Dense, Dropout, Reshape, Flatten,Add,MaxPool1D,BatchNormalization
import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

from keras.optimizers import SGD
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
gpus=tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
import random
random.seed(10)
np.random.seed(10)


class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass




class identity_block(Layer):
    def __init__(self, filter,kernel_size):
        super(identity_block, self).__init__()
        self.conv1=Conv1D(filters=filter[0],kernel_size=kernel_size,
                            strides=1,padding="same")
        self.conv2=Conv1D(filters=filter[1],
                            kernel_size=kernel_size,padding="same")
        self.conv3=Conv1D(filters=filter[2],
                                kernel_size=kernel_size,
                                padding="same")

        self.act1=Activation("relu")
        self.act2=Activation("relu")
        self.act3=Activation("relu")

    def call(self, x):
        first_layer =   x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x) 
        x = self.act2(x)
        x = self.conv3(x)
        residual =      Add()([x, first_layer])
        x =             self.act3(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1'       : self.conv1,
            'conv2'       : self.conv2,
            'conv3': self.conv3,
            'act1': self.act1,
            'act2': self.act2,
            'act3': self.act3,

        })
        return config



class convolution_block(Layer):
    def __init__(self, filter,kernel_size):
        super(convolution_block, self).__init__()
        self.conv1=Conv1D(filters=filter[0],kernel_size=kernel_size,
                            strides=1,padding="same")
        self.conv2=Conv1D(filters=filter[1],
                            kernel_size=kernel_size,padding="same")
        self.conv3=Conv1D(filters=filter[2],
                                kernel_size=kernel_size,
                                padding="same")
        self.conv4=Conv1D(filters=filter[2],
                                kernel_size=kernel_size,
                                padding="same")
        self.act1=Activation("relu")
        self.act2=Activation("relu")
        self.act3=Activation("relu")
        self.act4=Activation("relu")

    def call(self, x):
        first_layer =   x
        x =             self.conv1(x)
        x =             self.act1(x)
        x =             self.conv2(x)
        x =             self.act2(x)
        x =             self.conv3(x)
        x =             self.act3(x)

        first_layer =   self.conv4(first_layer)

        convolution =   Add()([x, first_layer])
        x =             self.act4(convolution)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1'       : self.conv1,
            'conv2'       : self.conv2,
            'conv3': self.conv3,
            'conv4': self.conv4,
            'act1': self.act1,
            'act2': self.act2,
            'act3': self.act3,
            'act4': self.act4,
        })
        return config

class ResNet(Disaggregator):

    def __init__(self, params):

        self.MODEL_NAME = "ResNet"
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',299)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size',512)
        self.load_model_path=params.get('load_model_path',None)
        self.appliance_params = params.get('appliance_params',{})
        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)

    def partial_fit(self,train_main,train_appliances,do_preprocessing=True,**load_kwargs):

        print("...............ResNet partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        train_main = pd.concat(train_main,axis=0)
        train_main = train_main.values.reshape((-1,self.sequence_length,1))
        
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs,axis=0)
            app_df_values = app_df.values.reshape((-1,self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances
        print(train_appliances)
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = 'ResNet-temp-weights-'+str(random.randint(0,100000))+'.h5'
                    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15,random_state=10)
                    history=model.fit(train_x,train_y,validation_data=(v_x,v_y),epochs=self.n_epochs,callbacks=[checkpoint],batch_size=self.batch_size)
                    model.load_weights(filepath)



    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):

        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_mains_df in test_main_list:

            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))

            for appliance in self.models:

                prediction = []
                model = self.models[appliance]
                prediction = model.predict(test_main_array ,batch_size=self.batch_size)

                #####################
                # This block is for creating the average of predictions over the different sequences
                # the counts_arr keeps the number of times a particular timestamp has occured
                # the sum_arr keeps the number of times a particular timestamp has occured
                # the predictions are summed for  agiven time, and is divided by the number of times it has occured
                
                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                for i in range(len(prediction)):
                    sum_arr[i:i + l] += prediction[i].flatten()
                    counts_arr[i:i + l] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]

                #################
                prediction = self.appliance_params[appliance]['mean'] + (sum_arr * self.appliance_params[appliance]['std'])
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

        return test_predictions




    def return_network(self):

        num_filters=30
        model = Sequential()

        model.add(ZeroPadding1D(padding=3,input_shape=(self.sequence_length,1)))        
        model.add(Conv1D(num_filters,48,activation="relu",strides=2))
        model.add(BatchNormalization(axis=2))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(3,strides=2))

        #Two types of residual block used for resnet
        model.add(convolution_block([num_filters,num_filters,num_filters],24))
        model.add(identity_block([num_filters,num_filters,num_filters],12))    
        model.add(identity_block([num_filters,num_filters,num_filters],6))

        #Fully connected layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(self.sequence_length,activation='linear'))

        model.summary()
        model.compile(loss='mse', optimizer='adam')


        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == 'train':            
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):

                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                    app_min=self.appliance_params[app_name]['min']
                    app_max=self.appliance_params[app_name]['max']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()


                processed_app_dfs = []
                for app_df in app_df_lst:                    
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])                    
                    new_app_readings = (new_app_readings - app_mean) / app_std  # /self.max_val
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))
                    
                    
                appliance_list.append((app_name, processed_app_dfs))
                #new_app_readings = np.array([ new_app_readings[i:i+n] for i in range(len(new_app_readings)-n+1) ])
                #print (new_mains.shape, new_app_readings.shape, app_name)

            return processed_mains_lst, appliance_list

        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                #new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_appliance_params(self,train_appliances):

        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            app_max=np.max(l)
            app_min=np.min(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std,'max':app_max,'min':app_min}})
