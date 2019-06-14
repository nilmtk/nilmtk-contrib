from .dataset import DataSet
from .metergroup import MeterGroup
import pandas as pd
<<<<<<< HEAD
from nilmtk.disaggregate import CombinatorialOptimisation, Mean, FHMM, Zero, Hart85, DAE
=======
from nilmtk.disaggregate import CombinatorialOptimisation, Mean, FHMM, Zero, DAE
>>>>>>> master
from nilmtk.disaggregate import Disaggregator
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
		self.train_store=None
		self.test_store=None
		self.experiment(d)

	
	def experiment(self,d):
		
		self.initialise(d)
<<<<<<< HEAD
		if 'train_store' in d:
			self.store_preprocessed_data()
			self.load_train_preprocessed_data()
		
		#self.load_datasets()
		# self.call_partial_fit()
=======
		self.store_datasets()
>>>>>>> master


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
		self.train_store=d['train_store']


	def store_preprocessed_data(self):



		# store train_mains chunk wise 
		store=pd.HDFStore(self.train_store,"w")
		d=self.train_datasets_dict
		for dataset in d:
			key='/train'
			print("Loading data for ",dataset, " dataset")
			key=key+"/"+dataset
			train=DataSet(d[dataset]['path'])
			for building in d[dataset]['buildings']:
				key=key+"/"+str(building)+"/mains/"
				print("Loading building ... ",building)
				train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				for i,chunk in enumerate(train.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period,chunksize=10000)):
					store[(key+"chunk"+str(i+1))]=chunk

		

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
					for i,chunk in enumerate(train.buildings[building].elec.submeters().select_using_appliances(type=appliance).load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period,chunksize=10000)):
						key='/train'+'/'+dataset+'/'+str(building)+'/'+appliance+'/chunk'+str(i+1)
						store[key]=chunk
				
		# store test_mains chunk wise
		d=self.test_datasets_dict

		for dataset in d:
			key='/test'
			print("Loading data for ",dataset, " dataset")
			key=key+"/"+dataset
			test=DataSet(d[dataset]['path'])
			for building in d[dataset]['buildings']:
				key=key+"/"+str(building)+"/mains/"
				print("Loading building ... ",building)
				test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				for i,chunk in enumerate(test.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period,chunksize=10000)):
					store[(key+"chunk"+str(i+1))]=chunk

		for appliance in self.appliances:
			
			test_df=pd.DataFrame()
			print("For appliance .. ",appliance)
			for dataset in d:
				key='/test'
				print("Loading data for ",dataset, " dataset")
				test=DataSet(d[dataset]['path'])
				for building in d[dataset]['buildings']:
					print("Loading building ... ",building)
					
					test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
					for i,chunk in enumerate(test.buildings[building].elec.submeters().select_using_appliances(type=appliance).load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period,chunksize=10000)):
						key='/test'+'/'+dataset+'/'+str(building)+'/'+appliance+'/chunk'+str(i+1)
						store[key]=chunk

		print(store.keys())
		store.close()		
		
	
	def load_train_preprocessed_data(self):
		
		store=pd.HDFStore(self.train_store,"r")

		# test for mean
		clf=Mean({})
		print(clf)
		for keys in store.keys():
			chunkno=1
			for appliance in self.appliances:
				if keys.find(appliance+"/chunk"+str(chunkno))>-1:
					self.train_submeters.append((appliance,store[keys]))									

		print(self.train_submeters)			
		clf.partial_fit(self.train_mains,self.train_submeters)			
		store.close()

		store=pd.HDFStore(self.train_store,"r")
		self.test_mains=store['/test/REDD/1/mains/chunk1']
		print("test mains ",self.test_mains)
		pred_overall={}
		#gt_overall,pred_overall['Mean']=self.predict(clf,self.test_mains,self.test_submeters, self.sample_period,'Europe/London')


	
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

				self.call_partial_fit()
				
	
	def update_method_dict(self):


		

		method_dict={}
		for i in self.method_dict:
			if i in self.methods:
				self.method_dict[i].update(self.methods[i])


	
		method_dict={'CO':CombinatorialOptimisation(self.method_dict['CO']),
					'FHMM':FHMM(self.method_dict['FHMM']),
					'Hart85':Hart85(self.method_dict['Hart85']),
					'DAE':DAE(self.method_dict['DAE']),
					'Mean':Mean(self.method_dict['Mean']),
					'Zero':Zero(self.method_dict['Zero'])
					}

		return method_dict


	
	def call_partial_fit(self):
		

		pred_overall={}
		classifiers=[]
<<<<<<< HEAD
		method_dict=self.update_method_dict()
=======
		method_dict={'CO':CombinatorialOptimisation(),'Mean':Mean(), 'FHMM': FHMM(), 'Zero': Zero(), 'DAE': DAE()}
>>>>>>> master
		
		# training models
		for name in self.methods:
			if name in method_dict:
				clf=method_dict[name]
				clf.partial_fit(self.train_mains,self.train_submeters)
				classifiers.append((name,clf))

		
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

	def store_test_mains(self,dataset,building):

		tempdf=pd.DataFrame()
		d=self.test_datasets_dict
		print("Loading building ... ",building)
		test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
		self.test_mains=(next(test.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)))

	
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
		chunk_size=5000
		rowcount=len(test_elec)
		strow,endrow=self.get_start_end_of_chunk(chunk_size,rowcount)
		print("strow ",strow)
		print("endrow ",endrow)
		
		# "ac_type" varies according to the dataset used. 
		# Make sure to use the correct ac_type before using the default parameters in this code.   
		
		for i in range(0,len(strow)):

			chunk=test_elec.iloc[strow[i]:endrow[i]]
			chunk_drop_na=chunk.dropna()
			if len(chunk_drop_na.columns) > 1:
				tempdf=pd.DataFrame()
				tempdf = clf.disaggregate_chunk(chunk_drop_na['power'][ac])
				pred[i]=tempdf
			else:
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
<<<<<<< HEAD


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
=======
>>>>>>> master
