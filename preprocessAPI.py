from nilmtk.dataset import DataSet
from nilmtk.metergroup import MeterGroup
import pandas as pd
from sklearn import preprocessing
from .disaggregate import CombinatorialOptimisation, Mean, FHMM, Zero
from .disaggregate import Disaggregator
from six import iteritems
from sklearn.metrics import mean_squared_error
import numpy as np

class PreprocessAPI():

	def __init__(self,d):
		self.power = {}
		self.sample_period=1
		self.appliances=[]
		self.name='neural-nilm'
		self.chunk_size = 1000
		self.train_store = 'preprocessed-train.h5'
		self.test_store = 'preprocessed-test.h5'
		self.test_ground_truth_store = 'ground-truth-test.h5'
		self.preprocessing_function = ['normalise','mean_scaling']
		self.train_datasets_dict={}
		self.test_datasets_dict={}
		self.train_submeters=[]
		self.train_mains=pd.DataFrame()
		self.test_submeters=[]
		self.test_mains=pd.DataFrame()
		
		self.experiment(d)

	
	def experiment(self,d):
		self.initialise(d)
		self.load_datasets()

	def initialise(self,d):
	
		for elems in d['params']['power']:
			self.power=d['params']['power']
		self.sample_period=d['sample_rate']
		self.name = d['name']
		for elems in d['appliances']:
			self.appliances.append(elems)
		self.chunk_size=d['chunk_size']
		self.train_datasets_dict=d['train']['datasets']
		self.test_datasets_dict=d['test']['datasets']
		self.train_store=d['train_store']
		self.test_store=d['test_store']
		self.preprocessing_function=d['preprocessing_function']
		self.sequence_length = 100
		self.max_val = 1500
		# Taking a default Max value
		self.max_powers = [self.max_val] * len(self.appliances)
		self.window_lengths = [self.sequence_length] * len(self.appliances)

		if 'sequence_length' in d:
			self.sequence_length = d['sequence_length']

		if 'max_val' in d:
			self.max_val = d['max_val']
		
		if 'max_powers' in d:
			self.max_powers = d['max_powers']

		if 'window_lengths' in d:
			self.window_lengths = d['window_lengths']

		if 'means' in d:
			self.means = d['means']

		if 'stds' in d:
			self.stds = d['std']

		if 'mean_mains' in d:
			self.mean_mains = d['mean_mains']

		if 'std_mains' in d:
			self.std_mains = d['std_mains']


	def load_datasets(self):

		d=self.train_datasets_dict

		print("............... Loading Data for preprocessing ...................")
		# store the train_main readings for all buildings
		
		print("............... Loading Train_Mains for preprocessing ...................")

		for dataset in d:
			print("Loading data for ",dataset, " dataset")
			
			for building in d[dataset]['buildings']:
					train=DataSet(d[dataset]['path'])
					print("Loading building ... ",building)
					train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
					mains_iterator = train.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)
					print (self.appliances)
					appliance_iterators = [train.buildings[building].elec.select_using_appliances(type=app_name).load(chunksize = self.chunk_size, physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period) for app_name in self.appliances]
					for chunk_num,chunk in enumerate (train.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)):
						#Dummry loop for executing on outer level. Just for looping till end of a chunk
						train_df = next(mains_iterator)
						train_df = train_df.dropna(axis=0)
						appliance_readings = []
						ix = train_df.index
						for i in appliance_iterators:
							appliance_df = next(i)
							appliance_df = appliance_df.dropna(axis=0)
							appliance_readings.append(appliance_df)
							ix = ix.intersection(appliance_df.index)

						train_df = train_df.loc[ix]	# Choosing the common timestamps


						for i in range(len(appliance_readings)):
							appliance_readings[i] = appliance_readings[i].loc[ix] # Choosing the Common timestamps

						train_appliances = []

						for cnt,i in enumerate(appliance_readings):
							train_appliances.append((self.appliances[cnt],i))

						train_df, train_appliances = self.preprocess_HDF5(train_df, train_appliances,method='train')

						self.store_preprocessed_data('train', train_df, train_appliances, dataset, building, chunk_num)


		print("...............Finished Loading Train mains and Appliances for preprocessing ...................")

		# store train submeters reading

		d=self.test_datasets_dict

		# store the test_main readings for all buildings

		for dataset in d:
			print("Loading data for ",dataset, " dataset")
			for building in d[dataset]['buildings']:
				test=DataSet(d[dataset]['path'])
				test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				mains_iterator = test.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)
				#print (self.appliances)
				appliance_iterators = [test.buildings[building].elec.select_using_appliances(type=app_name).load(chunksize = self.chunk_size, physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period) for app_name in self.appliances]
				for chunk_num,chunk in enumerate (test.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)):

					test_df = next(mains_iterator)
					test_df = test_df.dropna(axis=0)
					appliance_readings = []

					ix = test_df.index
					for i in appliance_iterators:
						appliance_df = next(i)
						appliance_df = appliance_df.dropna(axis=0)
						appliance_readings.append(appliance_df)
						ix = ix.intersection(appliance_df.index)

					test_df = test_df.loc[ix]	# Choosing the common timestamps


					for i in range(len(appliance_readings)):
						appliance_readings[i] = appliance_readings[i].loc[ix] # Choosing the Common timestamps

					test_appliances = []

					for cnt,i in enumerate(appliance_readings):
						test_appliances.append((self.appliances[cnt],i))

					test_df= self.preprocess_HDF5(test_df, submeters=None,method='test')

					print (test_df.shape, test_appliances[0][1].shape)

					self.store_preprocessed_data('test', test_df, test_appliances, dataset, building, chunk_num)





			
	def preprocess_HDF5(self, mains, submeters=None,method='train'):

		for function_name in self.preprocessing_function:
			
			if function_name =='seq2point':
				if method == 'train':
					aggregate_list = []
					appliance_list = []

					for app_index, (app_name,app_df) in enumerate(submeters):
						n = self.window_lengths[app_index]
						mean_appliance = self.means[app_index]
						std_appliance  = self.stds[app_index]
						units_to_pad = self.window_lengths[app_index]//2

						new_mains = mains.values.flatten()
						new_app_readings = app_df.values.reshape((-1,1))
						new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
						#new_app_readings = np.pad(new_app_readings, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))												

						# This is for choosing windows

						new_mains = np.array([ new_mains[i:i+n] for i in range(len(new_mains)-n+1) ])
						new_mains = (new_mains - self.mean_mains)/self.std_mains
						new_app_readings = (new_app_readings - mean_appliance)/std_appliance
						new_mains = pd.DataFrame(new_mains)
						new_app_readings = pd.DataFrame(new_app_readings)
						aggregate_list.append(new_mains)
						appliances_list((app_name, new_app_readings ))


						#new_app_readings = np.array([ new_app_readings[i:i+n] for i in range(len(new_app_readings)-n+1) ])



						#print (new_mains.shape, new_app_readings.shape, app_name)





			if function_name == 'neural-nilm':
				sequence_length  = self.sequence_length
				max_val = self.max_val
				if method=='train':
					print ("Training processing")
					
					mains = self.neural_nilm_preprocess_input(self.neural_nilm_choose_windows(mains.values, sequence_length),sequence_length,max_val)
					mains = mains.reshape((-1,sequence_length))
					mains_df = pd.DataFrame(mains)

					tuples_of_appliances = []
					for (appliance_name,df) in submeters:
						data = self.neural_nilm_preprocess_output(self.neural_nilm_choose_windows(df.values,sequence_length),sequence_length,max_val)
						data = data.reshape((-1,sequence_length))
						appliance_df  = pd.DataFrame(data)
						tuples_of_appliances.append((appliance_name, appliance_df))

					return mains_df, tuples_of_appliances

				if method=='test':

					mains = self.neural_nilm_preprocess_input(self.neural_nilm_choose_windows(mains.values, sequence_length),sequence_length,max_val)
					mains = mains.reshape((-1,sequence_length))
					mains_df = pd.DataFrame(mains)
					return mains_df

					#df_new = self.mean_scaling(train_mains)

	# def normalise(self, df):


	# 	float_array = df.values.astype(float)
	# 	min_max_scaler = preprocessing.MinMaxScaler()
	# 	scaled_array = min_max_scaler.fit_transform(float_array)
	# 	df_normalized = pd.DataFrame(scaled_array)
	# 	return df_normalized


	def neural_nilm_choose_windows(self, data, sequence_length):

		excess_entries =  sequence_length - (data.size % sequence_length)		
		lst = np.array([0] * excess_entries)
		arr = np.concatenate((data.flatten(), lst),axis=0)		
		return arr.reshape((-1,sequence_length))
	
	
		
	def neural_nilm_preprocess_input(self,windowed_x,sequence_length, max_value_of_reading):

		mean_sequence = np.mean(windowed_x,axis=1).reshape((-1,1))
		windowed_x = windowed_x - mean_sequence # Mean centering each sequence
		return (windowed_x/max_value_of_reading).reshape((-1,sequence_length))


	def neural_nilm_preprocess_output(self,windowed_y,sequence_length, max_value_of_reading):

		#self.appliance_wise_max[appliance_name] = self.default_max_reading
		return (windowed_y/max_value_of_reading).reshape((-1,sequence_length))


	def store_preprocessed_data(self, mode, mains_df, appliances_list, dataset_name, building_name, chunk_num):

		print ("Storing data for dataset %s building %s chunk_num %s" %(dataset_name,building_name,chunk_num))

		common_key = "/"+ str(dataset_name)+"/"+str(building_name)+"/"+str(chunk_num)+"/"


		if mode=='train':
			store = pd.HDFStore(self.train_store,"a")
			store.put(common_key+"mains", mains_df)

			for app_name, app_df in appliances_list:
				store.put(common_key+app_name, app_df)
			store.close()

		else:

			# Store the Processed Aggregate
			store = pd.HDFStore(self.test_store,"a")
			store.put(common_key+"mains", mains_df)
			store.close()		
			store = pd.HDFStore(self.test_ground_truth_store,'a')

			for app_name, app_df in appliances_list:
				store.put(common_key+app_name, app_df)
			store.close()		

