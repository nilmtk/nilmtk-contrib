from nilmtk.dataset import DataSet
from nilmtk.metergroup import MeterGroup
import pandas as pd
from .disaggregate import CombinatorialOptimisation, Mean, FHMM, Zero, DAE
from .disaggregate import Disaggregator
from six import iteritems
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class API():

	def __init__(self,d):

		self.chunk_size=40000
		self.power = {}
		self.sample_period = 1
		self.appliances = []
		self.methods = {}
		self.method_dict={'CO':{},'FHMM':{},'Hart85':{},'DAE':{},'Mean':{},'Zero':{}}
		self.pre_trained = False
		self.metrics = []
		self.train_datasets_dict = {}
		self.test_datasets_dict = {}
		self.train_submeters = []
		self.train_mains = pd.DataFrame()
		self.test_submeters = []
		self.test_mains = pd.DataFrame()
		self.gt_overall = {}
		self.pred_overall = {}
		self.classifiers=[]
		self.dictionary={}
		self.experiment(d)

	
	def experiment(self,d):
		
		self.dictionary=d
		self.initialise(d)
		if d['preprocessing']:
			print ('oo Training')
			self.train_test_preprocessed_data()
		else:
			self.load_datasets()
		


	def initialise(self,d):
	
		for elems in d['params']['power']:
			self.power = d['params']['power']
		self.sample_period = d['sample_rate']
		for elems in d['appliances']:
			self.appliances.append(elems)
		self.pre_trained = d['pre_trained']
		self.train_datasets_dict = d['train']['datasets']
		self.test_datasets_dict = d['test']['datasets']
		self.methods = d['methods']
		self.metrics = d['test']['metrics']


	
	def train_test_preprocessed_data(self):
		
		# chunkwise training and testing from preprocessed file
		# Training
		self.store_classifier_instances()

		d=self.dictionary

		#self.store_classifier_instances()

		train_file=pd.HDFStore(d['preprocess_train_path'],"r")
		
		keys=train_file.keys()

		
		# Processed HDF5 keys will be of the following format /dataset_name/building_name/

		
			 
		tuples = [i.split('/')[1:] for i in keys]
		datasets_list = list(set([i[0] for i in tuples]))

		for dataset_name in datasets_list:
			# Choose the buildings for the selected dataset
			available_buildings_in_current_dataset = list(set([i[1] for i in tuples if i[0]==dataset_name]))

			for building_id in available_buildings_in_current_dataset:
				available_chunks = list(set([i[2] for i in tuples if (i[0]==dataset_name) and i[1]==building_id]))

				for chunk_id in available_chunks:
					#print ()
					mains_df = train_file['/%s/%s/%s/%s' %(dataset_name,building_id,chunk_id,'mains')]
					list_of_appliances = []

					for app_name in self.appliances:
						appliance_df = train_file['/%s/%s/%s/%s' %(dataset_name,building_id,chunk_id,app_name)]
						list_of_appliances.append((app_name, appliance_df))

					self.train_mains  = mains_df
					self.train_submeters = list_of_appliances
					print ("Training on ",dataset_name,building_id,chunk_id,app_name)
					self.call_partial_fit()
			

		train_file.close()
		
		# testing
		
		test_mains=pd.HDFStore(self.dictionary['preprocess_test_path'])
		ground_truth=pd.HDFStore(self.dictionary['ground_truth'])
		

		keys = test_mains.keys()

		tuples = [i.split('/') for i in keys]

		for i in tuples:
			dataset = i[1]
			building = i[2]
			chunk_num = i[3]
			mains_df = test_mains['/%s/%s/%s/%s' % (dataset,building,chunk_num,'mains')]

			gt_overall = {}
			for app_name in self.appliances:

				gt_appliance = ground_truth['/%s/%s/%s/%s' % (dataset,building,chunk_num,app_name)]
				index = gt_appliance.index
				gt_appliance = gt_appliance.values.flatten()
				gt_appliance = pd.Series(gt_appliance,index=index)
				gt_overall[app_name] = gt_appliance

			gt_overall = pd.DataFrame(gt_overall)

			#print ("GT shape",gt_overall.shape)

			#print (gt_overall)

			pred_overall = {}

			for name, clf in self.classifiers:
				clf_prediction = {}

				prediction = clf.disaggregate_chunk(mains_df)
				#print (prediction)
				for app_name in self.appliances:
					index = gt_overall.index
					# Now, take the predictions. Sometimes predictions can have more values in the end. So get rid of extra vals in the end
					prediction_appliance = prediction[app_name].values.flatten()[:len(index)]
  					prediction_appliance = pd.Series(prediction_appliance,index=index)
						
					clf_prediction[app_name] = prediction_appliance

				pred_overall[name] = pd.DataFrame(clf_prediction)

			if len(gt_overall)>0:
				print ("\n\n")
				print ("Dataset: %s Building %s Chunk %s" % (dataset,building,chunk_num))
				for metrics in self.metrics:
					if metrics=='rmse':
						rmse = {}
						for clf_name,clf in self.classifiers:
							#print (gt_overall.shape, pred_overall[clf_name].shape, type(gt_overall),type(pred_overall[clf_name]))
							rmse[clf_name] = self.compute_rmse(gt_overall, pred_overall[clf_name])
						rmse = pd.DataFrame(rmse)
						print("............ " ,metrics," ..............")

						print(rmse)	

					if metrics=='mae':
						mae={}
						for clf_name,clf in self.classifiers:
							#print (gt_overall.shape, pred_overall[clf_name].shape, type(gt_overall),type(pred_overall[clf_name]))

							#print (gt_overall)
							#print (pred_overall[clf_name])
							mae[clf_name] = self.compute_mae(gt_overall, pred_overall[clf_name])
						mae = pd.DataFrame(mae)
						print("............ " ,metrics," ..............")
						print(mae)
		
		test_mains.close()
		ground_truth.close()



					# Raktim, prediction_appliance has the final prediction for the appliance. It doesn't have the extra zeros
					# The prediction is for app_name
					# Now, please find the errors 
					# The whohe thing is chunk wise
					# Code below is yours

				#

		# for dataset in datasets:
		# 	for building in buildings:
		# 		path='/test/'+dataset+'/'+building+'/'
		# 		chunkno=1
				
		# 		while 1:
		# 			pred_overall={}
		# 			chunk_mains_str=path+'chunk'+str(chunkno)+'/mains'
		# 			if chunk_mains_str not in test_mains_keys:
		# 				break
		# 			self.test_mains=test_mains[chunk_mains_str].dropna()
		# 			print(".........For key..........",chunk_mains_str)
					
		# 			# load test_appliances chunk wise
		# 			self.test_submeters=[]
		# 			for appliance in self.appliances:
		# 				chunk_appliances_str=path+'chunk'+str(chunkno)+'/'+appliance
		# 				self.test_submeters.append((appliance,ground_truth[chunk_appliances_str]))
		# 			for name,clf in self.classifiers:
		# 				print("............. For classifier ....... ",name)
		# 				gt_overall,pred_overall[name]=self.predict(clf,self.test_mains,self.test_submeters, self.sample_period,'Europe/London')			
		# 			chunkno +=1	
					
		# 			for metrics in self.metrics:
		# 				if metrics=='rmse':
		# 					rmse = {}
		# 					for clf_name,clf in self.classifiers:
		# 						rmse[clf_name] = self.compute_rmse(gt_overall, pred_overall[clf_name])
		# 					rmse = pd.DataFrame(rmse)
		# 					print("............ " ,metrics," ..............")
		# 					print(rmse)	

		# 				if metrics=='mae':
		# 					mae={}
		# 					for clf_name,clf in self.classifiers:
		# 						mae[clf_name] = self.compute_mae(gt_overall, pred_overall[clf_name])
		# 					mae = pd.DataFrame(mae)
		# 					print("............ " ,metrics," ..............")
		# 					print(mae)


	def load_datasets(self):


		d=self.train_datasets_dict
		
		print("............... Loading Data for training ...................")
		
		# store the train_main readings for all buildings
		for dataset in d:
			print("Loading data for ",dataset, " dataset")
			train=DataSet(d[dataset]['path'])
			for building in d[dataset]['buildings']:
				print("Loading building ... ",building)
				train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				self.train_mains=self.train_mains.append(next(train.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)))		
				
		

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
		

		
		
		# create instance of the training methods
		self.store_classifier_instances()
		# train models
		self.call_partial_fit()

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

				self.call_predict(self.classifiers)
				
	
	def store_classifier_instances(self):


		

		method_dict={}
		for i in self.method_dict:
			if i in self.methods:
				self.method_dict[i].update(self.methods[i])


		#print (self)
		# method_dict={'CO':CombinatorialOptimisation(self.method_dict['CO']),
		# 			'FHMM':FHMM(self.method_dict['FHMM']),
		# 			'DAE':DAE(self.method_dict['DAE']),
		# 			'Mean':Mean(self.method_dict['Mean']),
		# 			'Zero':Zero(self.method_dict['Zero'])
		# 			}

		method_dict = {'DAE': DAE(self.method_dict['DAE'])}
		for name in self.methods:
			if name in method_dict:
				clf=method_dict[name]
				self.classifiers.append((name,clf))


	
	def call_predict(self,classifiers):
		
		pred_overall={}
		for name,clf in classifiers:
			gt_overall,pred_overall[name]=self.predict(clf,self.test_mains,self.test_submeters, self.sample_period,'Europe/London')

		self.gt_overall=gt_overall
		self.pred_overall=pred_overall

		# metrics

		for metrics in self.metrics:
			if metrics=='rmse':
				rmse = {}
				for clf_name,clf in classifiers:
					rmse[clf_name] = self.compute_rmse(gt_overall, pred_overall[clf_name])
				rmse = pd.DataFrame(rmse)
				print("............ " ,metrics," ..............")
				print(rmse)	

			if metrics=='mae':
				mae={}
				for clf_name,clf in classifiers:
					mae[clf_name] = self.compute_mae(gt_overall, pred_overall[clf_name])
				mae = pd.DataFrame(mae)
				print("............ " ,metrics," ..............")
				print(mae)					

	def call_partial_fit(self):
		
		print ("Called Partial fit")
		
		# training models
		for name,clf in self.classifiers: 
			print (name,clf)
			clf.partial_fit(self.train_mains,self.train_submeters)
		

	
	def get_start_end_of_chunk(self,chunk_size,rowcount):

		currind=0
		strow=[]
		endrow=[]
		if rowcount<=chunk_size:
			strow.append(0)
			endrow.append(rowcount)
			currind +=rowcount
		while currind+chunk_size<rowcount:
			strow.append(currind)
			endrow.append(currind+chunk_size)
			currind +=chunk_size
		if(rowcount-currind > 100):
			strow.append(currind)
			endrow.append(currind+chunk_size)

		return strow,endrow

	def predict(self, clf, test_elec, test_submeters, sample_period, timezone):
		
		"""
		test_main: pd.DataFrame with pd.DatetimeIndex as index and 1 or more power columns
		returns: [train_appliance_1, train_appliance_i, ..., train_appliance_n]
		"""
		
		
		
		pred = {}
		gt= {}
		chunk_size=10000
		rowcount=len(test_elec)
		strow,endrow=self.get_start_end_of_chunk(chunk_size,rowcount)
		
		# "ac_type" varies according to the dataset used. 
		# Make sure to use the correct ac_type before using the default parameters in this code.   
		
		for i in range(0,len(strow)):

			chunk=test_elec.iloc[strow[i]:endrow[i]]
			chunk_drop_na=chunk.dropna()
			pred[i] = clf.disaggregate_chunk(chunk_drop_na)
			gt[i]={}

			for meter,data in test_submeters:
				gt[i][meter] = data.iloc[strow[i]:endrow[i]]
			gt[i] = pd.DataFrame({k:v.squeeze() for k,v in iteritems(gt[i]) if len(v)}, index=next(iter(gt[i].values())).index).dropna()

		# If everything can fit in memory

		gt_overall = pd.concat(gt)
		gt_overall.index = gt_overall.index.droplevel()
		pred_overall = pd.concat(pred)
		pred_overall.index = pred_overall.index.droplevel()

		# Having the same order of columns
		gt_overall = gt_overall[pred_overall.columns]

		#Intersection of index
		gt_index_utc = gt_overall.index.tz_convert("UTC")
		pred_index_utc = pred_overall.index.tz_convert("UTC")
		common_index_utc = gt_index_utc.intersection(pred_index_utc)

		common_index_local = common_index_utc.tz_convert(timezone)
		gt_overall = gt_overall.loc[common_index_local]
		pred_overall = pred_overall.loc[common_index_local]
		appliance_labels = [m for m in gt_overall.columns.values]
		gt_overall.columns = appliance_labels
		pred_overall.columns = appliance_labels
		return gt_overall, pred_overall


	# metrics
	def compute_mae(self,gt,pred):

		mae={}
		for appliance in gt.columns:
			mae[appliance]=mean_absolute_error(gt[appliance],pred[appliance])
		return pd.Series(mae)


	def compute_rmse(self,gt, pred):
			

		rms_error = {}
		for appliance in gt.columns:
			rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance]))
		return pd.Series(rms_error)