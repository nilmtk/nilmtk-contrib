from .dataset import DataSet
from .metergroup import MeterGroup
import pandas as pd
from nilmtk.disaggregate import CombinatorialOptimisation, Mean, FHMM
from nilmtk.disaggregate import Disaggregator
from six import iteritems
from sklearn.metrics import mean_squared_error
import numpy as np

class API():

	def __init__(self,d):
		self.power={}
		self.sample_period=1
		self.appliances=[]
		self.methods=[]
		self.pre_trained=False
		self.train_datasets_dict={}
		self.test_datasets_dict={}
		self.train_submeters=[]
		self.train_mains=pd.DataFrame()
		self.test_submeters=[]
		self.test_mains=pd.DataFrame()
		self.gt_overall = {}
		self.pred_overall = {}
		self.experiment(d)

	
	def experiment(self,d):
		self.initialise(d)
		self.store_datasets()
		# self.call_partial_fit()


	def initialise(self,d):
	
		for elems in d['params']['power']:
			self.power=d['params']['power']
		self.sample_period=d['sample_rate']
		for elems in d['appliances']:
			self.appliances.append(elems)
		self.pre_trained=d['pre_trained']
		self.train_datasets_dict=d['train']['datasets']
		self.test_datasets_dict=d['test']['datasets']
		self.methods=d['methods']
		
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
				
	def call_partial_fit(self):
		
		pred_overall={}
		classifiers=[]
		method_dict={'CO':CombinatorialOptimisation(),'Mean':Mean(), 'FHMM': FHMM()}
		
		# training models
		for i in self.methods:
			if i in method_dict:
				clf=method_dict[i]
				clf.partial_fit(self.train_mains,self.train_submeters)
				classifiers.append((i,clf))

		
		for name,clf in classifiers:
			gt_overall,pred_overall[name]=self.predict(clf,self.test_mains,self.test_submeters, self.sample_period,'Europe/London')

		self.gt_overall = gt_overall
		self.pred_overall = pred_overall

		rmse = {}
		
		for clf_name,clf in classifiers:
			rmse[clf_name] = self.compute_rmse(gt_overall, pred_overall[clf_name])
		rmse = pd.DataFrame(rmse)
		print("............RMSE .............")
		print(rmse)	

	def compute_rmse(self,gt, pred):
		    

	    rms_error = {}
	    for appliance in gt.columns:
	        rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance]))
	    return pd.Series(rms_error)


	def predict(self, clf, test_elec, test_submeters, sample_period, timezone):
        
		"""
		test_main: pd.DataFrame with pd.DatetimeIndex as index and 1 or more power columns
		returns: [train_appliance_1, train_appliance_i, ..., train_appliance_n]
		"""
		
		
		gt_overall1={}
		pred_overall1={}
		pred = {}
		gt= {}
		i=0
		currind=0
		strow=[]
		endrow=[]
		chunk_size=40000
		rowcount=len(test_elec)
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
		gt_overall = gt_overall.ix[common_index_local]
		pred_overall = pred_overall.ix[common_index_local]
		appliance_labels = [m for m in gt_overall.columns.values]
		gt_overall.columns = appliance_labels
		pred_overall.columns = appliance_labels
		return gt_overall, pred_overall