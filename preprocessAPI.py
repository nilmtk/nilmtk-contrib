from .dataset import DataSet
from .metergroup import MeterGroup
import pandas as pd
from sklearn import preprocessing
from nilmtk.disaggregate import CombinatorialOptimisation, Mean, FHMM, Zero
from nilmtk.disaggregate import Disaggregator
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
		self.preprocessing_function = ['normalise']
		self.train_datasets_dict={}
		self.test_datasets_dict={}
		self.train_submeters=[]
		self.train_mains=pd.DataFrame()
		self.test_submeters=[]
		self.test_mains=pd.DataFrame()
		
		self.experiment(d)

	
	def experiment(self,d):
		self.initialise(d)
		self.store_datasets()

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

	def store_datasets(self):

		d=self.train_datasets_dict

		print("............... Loading Data for training ...................")
		# store the train_main readings for all buildings
		for dataset in d:
			print("Loading data for ",dataset, " dataset")
			train=DataSet(d[dataset]['path'])
			for building in d[dataset]['buildings']:
					print("Loading building ... ",building)
					train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
					
					self.train_mains=self.train_mains.append(next(train.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)))		
		

		# store train submeters reading
		train_buildings=pd.DataFrame()
		for appliance in self.appliances:
			train_df=pd.DataFrame()
			print("For appliance .. ",appliance)
			for dataset in d:
				print("Loading data for ",dataset, " dataset")
				train=DataSet(d[dataset]['path'])
				for building in d[dataset]['buildings']:
					print("Loading building ... ",building)
					
					# store data for submeters
					train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
					train_df=train_df.append(next(train.buildings[building].elec.submeters().select_using_appliances(type=appliance).load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period)))
					

					# store data for mains
					
			self.train_submeters.append((appliance,train_df))	
		

		sequence_length = 100

		max_val = 5000


		d=self.test_datasets_dict

		# store the test_main readings for all buildings
		for dataset in d:
			print("Loading data for ",dataset, " dataset")
			test=DataSet(d[dataset]['path'])
			for building in d[dataset]['buildings']:
				test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				self.test_mains=(next(test.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)))		
				self.test_submeters=[]
				for appliance in self.appliances:
					test_df=next((test.buildings[building].elec.submeters().select_using_appliances(type=appliance).load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period)))
					self.test_submeters.append((appliance,test_df))

		for appliance, df in self.test_submeters:
				self.preprocess_HDF5(df)
		self.store_preprocessed_data()

	def preprocess_HDF5(self, df):
		print(df.head())
		for function_name in self.preprocessing_function:
			if function_name == 'normalise':
				df_new = self.normalise(df)
				print(df_new)

	def normalise(self, df):


		float_array = df.values.astype(float)
		min_max_scaler = preprocessing.MinMaxScaler()
		scaled_array = min_max_scaler.fit_transform(float_array)
		df_normalized = pd.DataFrame(scaled_array)
		return df_normalized


	def neural_nilm_choose_windows(self, data, sequence_length):

		excess_entries =  sequence_length - (data.size % sequence_length)		
		lst = np.array([0] * excess_entries)
		arr = np.concatenate((data.flatten(), lst),axis=0)		
		return arr.reshape((-1,sequence_length))
	
	
		
	def neural_nilm_preprocess_input(self,windowed_x,sequence_length, max_value_of_reading):

		mean_sequence = np.mean(windowed_x,axis=1).reshape((-1,1))
		windowed_x = windowed_x - mean_sequence # Mean centering each sequence
		return (windowed_x/max_value_of_reading).reshape((-1,sequence_length,1))


	def neural_nilm_preprocess_output(self,windowed_y,sequence_length, max_value_of_reading):

		#self.appliance_wise_max[appliance_name] = self.default_max_reading
		return (windowed_y/max_value_of_reading).reshape((-1,sequence_length,1))


	def store_preprocessed_data(self):
		# store train_mains chunk wise 

		sequence_length	 = 100

		max_val = 1500
		store=pd.HDFStore('train-pre.h5',"w")
		d=self.train_datasets_dict
		for dataset in d:
			
			print("Loading data for ",dataset, " dataset")
			
			train=DataSet(d[dataset]['path'])
			for building in d[dataset]['buildings']:
				key='/train'
				key=key+"/"+dataset
				key=key+"/"+str(building)+"/mains/"
				print("Loading building ... ",building)
				train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				for i,chunk in enumerate(train.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period,chunksize=self.chunk_size)):
					chunk_ = self.neural_nilm_preprocess_input(self.neural_nilm_choose_windows(chunk.values, sequence_length),sequence_length,max_val)
					
					print (chunk.shape)
					print (chunk_.shape)

					chunk_ =  pd.DataFrame(chunk_.flatten())

					store[(key+"chunk"+str(i+1))]=chunk_
					

		# store train_appliances chunk wise
		
		for appliance in self.appliances:
			
			train_df=pd.DataFrame()
			print("For appliance .. ",appliance)
			for dataset in d:
				key='/train'
				print("Loading data for ",dataset, " dataset")
				train=DataSet(d[dataset]['path'])
				for building in d[dataset]['buildings']:
					print("Loading building ... ",building)
					
					train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
					for i,chunk in enumerate(train.buildings[building].elec.submeters().select_using_appliances(type=appliance).load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period,chunksize=self.chunk_size)):
						chunk_ = self.neural_nilm_preprocess_output(self.neural_nilm_choose_windows(chunk.values,sequence_length),sequence_length,max_val)
						print (chunk.shape)
						print (chunk_.shape)

						chunk_ =  pd.DataFrame(chunk_.flatten())
						key='/train'+'/'+dataset+'/'+str(building)+'/'+appliance+'/chunk'+str(i+1)
						store[key]=chunk
				

		print(store.keys())
		store.close()		
		