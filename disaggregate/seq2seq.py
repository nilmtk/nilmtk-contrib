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


class Seq2Seq(Disaggregator):
	
	def __init__(self, d):

		
		self.sequence_length = 99
		self.n_epochs = 100
		self.trained = False
		self.models = OrderedDict()
		#self.max_value = 6000
		self.mains_mean = 1000
		self.mains_std = 1800
		self.batch_size = 512

		self.appliance_std = None

		if 'sequence_length' in d: 
			if d['sequence_length']%2==0:
				raise ValueError("Sequence length should be a odd number!!!")
			self.sequence_length = d['sequence_length']

		if 'n_epochs' in d: 
			self.n_epochs = d['n_epochs']

		if 'mains_mean' in d:
			self.mains_mean = d['mains_mean']

		if 'mains_std' in d:
			self.mains_std = d['mains_std']

		if 'appliance_params' in d:
			self.appliance_params = d['appliance_params']
		

	# def mse(self,y_true, y_pred):

	# 	return self.appliance_std*K.sqrt(K.mean((y_pred - y_true)**2))

	def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
		
		print("...............DAE partial_fit running...............")


		if do_preprocessing:
			train_main,train_appliances = self.call_preprocessing(train_main,train_appliances,'train')


		train_main = np.array([i.values.reshape((self.sequence_length,1)) for i in train_main])
		
		new_train_appliances  = []
		
		for app_name, app_df in train_appliances:
			app_df = np.array([i.values for i in app_df]).reshape((-1,self.sequence_length))
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
			
			if train_main.size>0:
				# Sometimes chunks can be empty after dropping NANS
				if len(train_main)>10:
					# Do validation when you have sufficient samples
					filepath = 'temp-weights.h5'
					#print (train_main.shape,power.shape)
					
					checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
					train_x,v_x,train_y,v_y = train_test_split(train_main,power,test_size=.15)
					#train_x = train_x[:1000]
					#train_y = train_y[:1000]
					model.fit(train_x,train_y,validation_data = [v_x,v_y],epochs = self.n_epochs, callbacks = [checkpoint], batch_size=self.batch_size)
					model.load_weights(filepath)

					# plt.plot(v_y.flatten()[:self.sequence_length]*self.max_val,color='r')
					# plt.plot(np.max(model.predict(v_x).flatten()[:self.sequence_length]*self.max_val,0),color='b')
					# plt.plot(v_x[self.sequence_length//2 - 1]*self.max_val,color='g')
					pred = model.predict(v_x)
# 					for i in range(10):
# 						plt.plot(v_y[i].flatten(),label='Truth')
# 						plt.plot(pred[i].flatten(),label='Pred')
# 						plt.legend()
# 						plt.show()

				#else:
					#model.fit(train_main, power, epochs = self.n_epochs)
				
			

	def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
		
		
		# t_ = pd.concat(test_main_list,axis=1)
		# print (type(test_main_list))
		# print ("Testing")
		# print (test_main_list[0].shape)
		# print (t_.shape)
		
		if do_preprocessing:
			
			test_main_list = self.call_preprocessing(test_main_list,submeters=None,method='test')
		
		# print (test_main_list[0].shape)
		# test_main_list = pd.concat(test_main_list,axis=1)
		# print (test_main_list.shape)
		# test_main = test_main_list.values.reshape((-1,self.sequence_length,1))
		
		# plt.figure()
		# plt.plot(range(self.sequence_length),test_main[0].flatten())
		# plt.plot(range(1,self.sequence_length+1),test_main[1].flatten())
		# plt.show()
		print ("New testing")
		#print (test_main_list[0].shape)
		#print (pd.concat(test_main_list,axis=1).shape)
		#print (pd.concat(test_main_list,axis=0).shape)
		#test_main_list = np.array([i.values for i in test_main_list])
		disggregation_dict = {}
		test_predictions = []

		#print (test_main_list.shape)
		print ("Length")
		print (len(test_main_list))
		test_main_array = np.array([window.values.flatten() for window in test_main_list])
		test_main_array = test_main_array.reshape((-1,self.sequence_length,1))
		print ("Max input")
		print (np.max(test_main_array),np.min(test_main_array))
		for appliance in self.models:

			prediction = []

			model = self.models[appliance]

			#for data in test_main_list:

			#	prediction.append(model.predict(data.values.reshape((-1,self.sequence_length,1))),batch_size=self.batch_size)

			prediction = model.predict(test_main_array,self.batch_size)
			
			l = self.sequence_length

			

			n = len(prediction) + l - 1
			val_arr = np.zeros((n))
			counts_arr = np.zeros((n))
			o = len(val_arr)
			for i in range(len(prediction)):
				val_arr[i:i+l]+=prediction[i].flatten()
				counts_arr[i:i+l]+=1
				
			for i in range(len(val_arr)):
				val_arr[i] = val_arr[i]/counts_arr[i]

			prediction = self.appliance_params[appliance]['mean'] + (val_arr * self.appliance_params[appliance]['std'])
			
			valid_predictions = prediction.flatten()
			
			valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
			
			df = pd.Series(valid_predictions)

			disggregation_dict[appliance] = df
		
		results = pd.DataFrame(disggregation_dict,dtype='float32')
			# plt.figure()
			# plt.plot(t_.values.flatten(),label='truth')
			# plt.plot(valid_predictions,label='prediction')
			# plt.title(appliance)
			# plt.legend()
			
			# plt.figure()
			# plt.plot(val_arr.flatten()[:1000])
			# print (test_main.shape)
			# print (valid_predictions.shape)
			# n = len(test_main) + l - 1
			# val_arr = np.zeros((n))
			# counts_arr = np.zeros((n))
			# o = len(val_arr)
			# for i in range(len(test_main)):
			# 	val_arr[i:i+l]+=test_main[i].flatten()
			# 	counts_arr[i:i+l]+=1
				
			# for i in range(len(val_arr)):
			# 	val_arr[i] = val_arr[i]/counts_arr[i]
				
				
			# plt.plot(val_arr.flatten()[:1000],label='input')
			# plt.legend()
		# plt.show()
		test_predictions.append(results)

		#print (test_predictions[-1])
   
		return test_predictions


	def return_network(self):
		
		model = Sequential()
		# 1D Conv
		model.add(Conv1D(30, 10, activation="relu", input_shape=(self.sequence_length, 1), strides=2))
		model.add(Conv1D(30, 8, activation='relu',strides = 2))
		model.add(Conv1D(40, 6, activation='relu',strides = 1))
		model.add(Conv1D(50, 5, activation='relu',strides = 1))
		model.add(Dropout(.2))
		model.add(Conv1D(50, 5, activation='relu',strides = 1))
		model.add(Dropout(.2))
		model.add(Flatten())
		model.add(Dense(1024,activation='relu'))
		model.add(Dropout(.2))
		model.add(Dense(self.sequence_length))
		#optimizer = SGD(lr=self.learning_rate)
		model.compile(loss='mse', optimizer='adam')#,metrics=[self.mse])

		#optimizer = SGD(lr=self.learning_rate)
		
		return model

	def call_preprocessing(self,mains,submeters,method):

		if method == 'train':
			mains = pd.concat(mains,axis=1)

			new_mains = mains.values.flatten()
			n =  self.sequence_length
			units_to_pad = n//2
			#new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
			new_mains = np.array([ new_mains[i:i+n] for i in range(len(new_mains)-n+1) ])
			new_mains = (new_mains - self.mains_mean)/self.mains_std
			mains_df_list = [pd.DataFrame(window) for window in new_mains]
			#new_mains = pd.DataFrame(new_mains)
			appliance_list = []
			for app_index, (app_name,app_df) in enumerate(submeters):

				if app_name in self.appliance_params:
					app_mean = self.appliance_params[app_name]['mean']
					app_std = self.appliance_params[app_name]['std']

				app_df = pd.concat(app_df,axis=1)
				#mean_appliance = self.means[app_index]
				#std_appliance  = self.stds[app_index]
				new_app_readings = app_df.values.flatten()
				#new_app_readings = np.pad(app_df.values.flatten(), (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
				new_app_readings = np.array([ new_app_readings[i:i+n] for i in range(len(new_app_readings)-n+1) ])
				
				#new_app_readings = np.pad(new_app_readings, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))												

				# This is for choosing windows

				
				#new_mains = (new_mains - self.mean_mains)/self.std_mains
				
				#new_app_readings = (new_app_readings - mean_appliance)/std_appliance
				new_app_readings = (new_app_readings - app_mean)/app_std#/self.max_val
				# I know that the following window has only one value
				app_df_list = [pd.DataFrame(window) for window in new_app_readings]
				#new_app_readings = pd.DataFrame(new_app_readings)
				#aggregate_list.append(new_mains)
				appliance_list.append((app_name, app_df_list ))
				#new_app_readings = np.array([ new_app_readings[i:i+n] for i in range(len(new_app_readings)-n+1) ])
				#print (new_mains.shape, new_app_readings.shape, app_name)
			
			return mains_df_list,appliance_list

		else:

			mains = pd.concat(mains,axis=1)

			new_mains = mains.values.flatten()
			n =  self.sequence_length
			units_to_pad = n//2
			#new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
			new_mains = np.array([ new_mains[i:i+n] for i in range(len(new_mains)-n+1) ])
			new_mains = (new_mains - self.mains_mean)/self.mains_std
			new_mains = new_mains.reshape((-1,self.sequence_length,1))
			print (" test New mains shape")
			print (new_mains.shape)
			mains_df_list = [pd.DataFrame(window) for window in new_mains]

			return mains_df_list




	  

# from __future__ import print_function, division
# from warnings import warn

# from nilmtk.disaggregate import Disaggregator
# from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import OrderedDict 

# from keras.optimizers import SGD
# from keras.models import Sequential
# import matplotlib.pyplot as  plt


# class DAE(Disaggregator):
	
#	 def __init__(self, sequence_length=100, learning_rate=1e-3, n_epochs = 10):
		
#		 self.learning_rate = learning_rate
#		 self.sequence_length = sequence_length
#		 self.n_epochs = n_epochs
#		 self.trained = False
#		 self.models = OrderedDict()
#		 self.std_of_random_sample = None
#		 self.appliance_wise_max = OrderedDict()
#		 self.default_max_reading = 10000
		
#	 def partial_fit(self, train_main, train_appliances, do_preprocessing = False, chunk_id=None, **load_kwargs):


#		 train_main = train_main.fillna(0)

#		 print (train_main.shape)

#		 new_train_appliances = []

#		 for index, (appliance_name, power) in enumerate(train_appliances):

#			 print (power.shape)

#			 new_train_appliances.append((appliance_name,power.fillna(0)))

#		 train_appliances = new_train_appliances + []

		
#		 print("...............DAE partial_fit running...............")

#		 if do_preprocessing:
			
#			 #print (type(train_main))
			
#			 #print (train_main.shape)
			
			
#			 train_main = self.preprocess_input(self.choose_windows(train_main.values))
			
#			 #print (train_main.shape)
			
#			 new_train_appliances = []
			
#			 for index, (appliance_name, power) in enumerate(train_appliances):
				
#				 # Modifying the power
#				 #print (power.shape)
				
#				 preprocessed_power = self.preprocess_output(self.choose_windows(power.values), appliance_name)
					
#				 new_train_appliances.append((appliance_name,preprocessed_power))
			
#			 train_appliances = new_train_appliances
			
#		 # If the data is already preprocessed
		
#		 for appliance_name, power in train_appliances:
			
#			 if appliance_name not in self.models:
#				 print ("New training for ",appliance_name)
#				 self.models[appliance_name] = self.return_network()
			
#			 else:
				
#				 print ("Retrained model for ",appliance_name)
				
#			 model = self.models[appliance_name]
#			 #print (train_main.shape)
#			 #print (power.shape)
			
#			 print (np.max(train_main))
#			 print (np.max(power))
			
#			 model.fit(train_main, power, epochs = self.n_epochs)

			
			

#	 def disaggregate_chunk(self, test_main, do_preprocessing=True):
		
#		 original_test_main = test_main.copy()
		
#		 if do_preprocessing:
			
#			 test_main = self.preprocess_input(self.choose_windows(test_main.values))
		
#		 disaggregated_power_dict = {}
		
#		 for appliance in self.models:
			
#			 prediction = self.models[appliance].predict(test_main)
			
#			 prediction = prediction * self.appliance_wise_max[appliance]
			
#			 valid_predictions = prediction.flatten()[:original_test_main.size]
			
#			 valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
			
#			 df = pd.Series(valid_predictions,index=original_test_main.index)

#			 disaggregated_power_dict[appliance] = df
#			 #tuples_of_predictions.append((appliance, df))

		
#		 return pd.DataFrame(disaggregated_power_dict,dtype='float32')
		

			
 
#	 def choose_windows(self, data):

#		 excess_entries =  self.sequence_length - (data.size % self.sequence_length)
		
#		 lst = np.array([0] * excess_entries)
#		 # print("lst...")
#		 # print(lst)
#		 # print(len(lst))
#		 # print (data.shape)
#		 arr = np.concatenate((data.flatten(), lst),axis=0)
#		 # print("arr...")
#		 # print(arr)
#		 # print(len(arr))
#		 return arr.reshape((-1,self.sequence_length))

		
#	 def preprocess_input(self,windowed_x):
		
#		 mean_sequence = np.mean(windowed_x,axis=1).reshape((-1,1))
		
#		 windowed_x = windowed_x - mean_sequence # Mean centering each sequence
		
# #		 if self.std_of_random_sample==None:

# #			 std_of_random_sample = 0

# #			 while std_of_random_sample==0:

# #				 random_index = np.random.randint(0,len(windowed_x))
				
# #				 std_of_random_sample = np.std(windowed_x[random_index])
			
			
# #			 self.std_of_random_sample = std_of_random_sample
		
		
#		 self.std_of_random_sample = self.default_max_reading

#		 print ( self.std_of_random_sample)
		
#		 return (windowed_x/self.std_of_random_sample).reshape((-1,self.sequence_length,1))
	
		
		
#	 def preprocess_output(self, windowed_y, appliance_name):
		
#		 if appliance_name not in self.appliance_wise_max:
			
#			 self.appliance_wise_max[appliance_name] = self.default_max_reading
		
#		 return (windowed_y/self.appliance_wise_max[appliance_name]).reshape((-1,self.sequence_length,1))
		
#	 def return_network(self):
		
#		 model = Sequential()

#		 # 1D Conv
#		 model.add(Conv1D(8, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
#		 model.add(Flatten())

#		 # Fully Connected Layers
#		 model.add(Dropout(0.2))
#		 model.add(Dense((self.sequence_length)*8, activation='relu'))

#		 model.add(Dropout(0.2))
#		 model.add(Dense(128, activation='relu'))

#		 model.add(Dropout(0.2))
#		 model.add(Dense((self.sequence_length)*8, activation='relu'))

#		 model.add(Dropout(0.2))

#		 # 1D Conv
#		 model.add(Reshape(((self.sequence_length), 8)))
#		 model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

#		 optimizer = SGD(lr=self.learning_rate)
		
#		 model.compile(loss='mse', optimizer='adam')

#		 return model
