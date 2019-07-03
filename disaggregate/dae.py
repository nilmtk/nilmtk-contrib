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



class DAE(Disaggregator):
    
    def __init__(self, d):

        self.sequence_length = 300
        self.n_epochs = 100
        self.trained = False
        self.models = OrderedDict()
        self.mains_mean = 1000
        self.mains_std = 1800
        self.batch_size = 512

        if 'sequence_length' in d: 
            self.sequence_length = d['sequence_length']

        if 'n_epochs' in d: 
            self.n_epochs = d['n_epochs']

        if 'mains_mean' in d:
            self.mains_mean = d['mains_mean']

        if 'mains_std' in d:
            self.mains_std = d['mains_std']

        if 'appliance_params' in d:
            self.appliance_params = d['appliance_params']
        
        
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True,**load_kwargs):
        
        print("...............DAE partial_fit running...............")

        print (train_main[0].shape)
        print (train_appliances[0][1][0].shape)
        print ("OK")
        #print (train_main[0])

        if do_preprocessing:
            train_main,train_appliances = self.call_preprocessing(train_main,train_appliances,'train')

        train_main = np.array([i.values.reshape((self.sequence_length,1)) for i in train_main])
        
        new_train_appliances  = []
        
        for app_name, app_df in train_appliances:
            app_df = np.array([i.values.reshape((self.sequence_length,1)) for i in app_df])
            new_train_appliances.append((app_name, app_df))
            
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
            
            if len(train_main)>3:
                # Do validation when you have sufficient samples

                filepath = 'temp-weights.h5'
                #print (train_main.shape)
                #print (power.shape)
                #print ()

                checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                train_x,v_x,train_y,v_y = train_test_split(train_main,power,test_size=.15)

                
                model.fit(train_x,train_y,validation_data = [v_x,v_y],epochs = self.n_epochs, callbacks = [checkpoint],shuffle=True,batch_size=self.batch_size)
                model.load_weights(filepath)

                pred = model.predict(v_x)
#                 for i in range(10):
#                         plt.plot(v_y[i].flatten(),label='Truth')
#                         plt.plot(pred[i].flatten(),label='Pred')
#                         plt.legend()
#                         plt.show()
                # v_pred = model.predict(v_x)

                # for i in range(len(v_x)):
                #     plt.plot(v_x[i].flatten()*self.mains_std,label='Input')
                #     plt.plot(v_y[i].flatten()*self.mains_std,label='Truth')
                #     plt.plot(v_pred[i].flatten()*self.mains_std,label='prediction')
                #     plt.legend()
                #     plt.show()

                # p = model.predict(train_main)
                # plt.figure(figsize=(20,8))
                # plt.plot(train_main.flatten()[:1000],label='Inp')
                # plt.plot(p.flatten()[:1000],label='pred')
                # plt.plot(power.flatten()[:1000],label='truth')
                # plt.legend()
                # plt.show()

                
                #h_id = 0

                # p = model.predict(v_x) * self.max_val
                # t = v_y *self.max_val
                # x = v_x*self.max_val

                # for  i in range(len(p)):
                #     plt.figure()
                #     plt.plot(t[i],color='g')
                #     plt.plot(p[i],color='r')
                #     plt.plot(x[i],color='b')
                #     plt.show()
                
            else:
                print ("This chunk has small number of samples, so skipping the training")
   

    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list,submeters=None,method='test')

        test_predictions = []

        for test_main in test_main_list:

            #print (np.max(test_main))

            test_main = test_main.values
            
            test_main = test_main.reshape((-1,self.sequence_length,1))
            
            disggregation_dict = {}
            
            # plt.figure(figsize=(20,8))
            #plt.figure()
            #plt.plot(test_main.flatten()*self.max_val,label='input')

            for appliance in self.models:
                
                prediction = self.models[appliance].predict(test_main)
                
                #prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                # print (self.mains_std)
                # print (np.mean(prediction))

                app_mean = self.appliance_params[appliance]['mean']
                app_std = self.appliance_params[appliance]['std']


                prediction = app_mean + (prediction * app_std)

                # print (np.mean(prediction))
               
                valid_predictions = prediction.flatten()
                
                valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
                
                df = pd.Series(valid_predictions)

                disggregation_dict[appliance] = df
                # plt.plot(test_main.flatten() * self.mains_std)
                # plt.plot(valid_predictions.flatten())
                # plt.show()
            
                #plt.plot(valid_predictions,label=appliance)

            #plt.legend()
            #plt.show()


            # plt.legend()
            # plt.show()
            results = pd.DataFrame(disggregation_dict,dtype='float32')

            test_predictions.append(results)

            #break

        #print (test_predictions[-1])
        # plt.title("test")
        # plt.figure(figsize=(20,8))
        # plt.plot(pd.concat(test_main_list).values.flatten()[:1000]*self.mains_std,color='r')
        # plt.plot(pd.concat(test_predictions).values.flatten()[:1000],color='g')

        # plt.show()
        return test_predictions
            

    def return_network(self):
        
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(8, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(Flatten())

        # Fully Connected Layers
        #model.add(Dropout(0.2))
        model.add(Dense((self.sequence_length)*8, activation='relu'))

        #model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))

        #model.add(Dropout(0.2))

        model.add(Dense((self.sequence_length)*8, activation='relu'))

        #model.add(Dropout(0.2))

        # 1D Conv
        model.add(Reshape(((self.sequence_length), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

        
        
        model.compile(loss='mse', optimizer='adam')

        return model

    def call_preprocessing(self, mains, submeters, method):
        sequence_length  = self.sequence_length

        #max_val = self.max_val
        if method=='train':
            print ("Training processing")
            print (mains[0].shape,submeters[0][1][0].shape)
            mains = pd.concat(mains,axis=1)
            mains = self.neural_nilm_preprocess_input(mains.values,sequence_length,self.mains_mean,self.mains_std,True)
            print ("Means is ")
            print (np.mean(mains))
            print (mains.shape,np.max(mains))
            mains_df_list = [pd.DataFrame(window) for window in mains]

            tuples_of_appliances = []

            for (appliance_name,df) in submeters:
                # if appliance_name in self.appliance_params:
                #     app_mean = 

                if appliance_name in self.appliance_params:
                    app_mean = self.appliance_params[appliance_name]['mean']
                    app_std = self.appliance_params[appliance_name]['std']

                df = pd.concat(df,axis=1)

                #data = self.neural_nilm_preprocess_output(df.values, sequence_length,app_mean,app_std,False)
                data = self.neural_nilm_preprocess_output(df.values, sequence_length,app_mean,app_std,True)
                
                appliance_df_list  = [pd.DataFrame(window) for window in data]

                tuples_of_appliances.append((appliance_name, appliance_df_list))

            return mains_df_list, tuples_of_appliances

        if method=='test':
            print ("Testing processing")
            mains = pd.concat(mains,axis=1)
            mains = self.neural_nilm_preprocess_input(mains.values ,sequence_length,self.mains_mean,self.mains_std,False)
            print ("Means is ")
            print (np.mean(mains))
            print (mains.shape,np.max(mains))
            mains_df_list = [pd.DataFrame(window) for window in mains]
            return mains_df_list

    # def neural_nilm_choose_windows(self, data, sequence_length):

    #     excess_entries =  sequence_length - (data.size % sequence_length)       
    #     lst = np.array([0] * excess_entries)
    #     arr = np.concatenate((data.flatten(), lst),axis=0)   

    #     return arr.reshape((-1,sequence_length))
    
    
        
    def neural_nilm_preprocess_input(self,data,sequence_length, mean, std, overlapping=False):

        #mean_sequence = np.mean(windowed_x,axis=1).reshape((-1,1))
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0)   
        if overlapping:
            windowed_x = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_x = arr.reshape((-1,sequence_length))

        windowed_x = windowed_x - mean
        #mean_sequence # Mean centering each sequence
        # print ("Just flat")
        # plt.plot(windowed_x.flatten()[:1000])
        # plt.ylim(0,2000)
        # plt.show()
        return (windowed_x/std).reshape((-1,sequence_length))


    def neural_nilm_preprocess_output(self,data,sequence_length, mean, std, overlapping=False):

        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0) 

        if overlapping:  
            windowed_y = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_y = arr.reshape((-1,sequence_length))

        

        #self.appliance_wise_max[appliance_name] = self.default_max_reading
        windowed_y = windowed_y - mean
        # plt.plot(windowed_y.flatten()[:1000])
        # plt.ylim(0,2000)
        # plt.show()
        return (windowed_y/std).reshape((-1,sequence_length))
        #return (windowed_y/max_value_of_reading).reshape((-1,sequence_length))

