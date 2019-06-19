from __future__ import print_function, division
from warnings import warn

import pandas as pd
import numpy as np

from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class Zero(Disaggregator):

	def __init__(self, d):
		self.model=[]
		self.MIN_CHUNK_LENGTH = 100
		self.MODEL_NAME = 'Zero'

	def partial_fit(self, train_main, train_appliances, **load_kwargs):

		'''
			train_main :- pandas DataFrame. It will contain the mains reading.
			train_appliances :- [('appliance1',df1),('appliance2',df2),...]

		'''
		train_main = pd.concat(train_main, axis=0)
		train_app_tmp = []

		for app_name, df_list in train_appliances:
			df_list = pd.concat(df_list, axis=0)
			train_app_tmp.append((app_name,df_list))

		train_appliances = train_app_tmp

		print("...............Zero partial_fit running...............")
		for appliance,readings in train_appliances:
			
			# there will be only off state for all appliances. 
			# the algorithm will always predict zero

			self.model.append({
				'states': 0,
				'training_metadata': appliance
				})

	def disaggregate_chunk(self,test_mains):
		print("...............Zero disaggregate_chunk running...............")
		if len(test_mains) < self.MIN_CHUNK_LENGTH:
			raise RuntimeError("Chunk is too short")

		appliance_powers_dict={}
		for i,model in enumerate(self.model):
			print("Estimating power demand for '{}'"
                  .format(model['training_metadata']))
			# a list of predicted power values for ith appliance
			predicted_power=[self.model[i]['states'] for j in range(0,test_mains.shape[0])]
			column=pd.Series(predicted_power,index=test_mains.index,name=i)
			appliance_powers_dict[self.model[i]['training_metadata']]=column

		appliance_powers=pd.DataFrame(appliance_powers_dict,dtype='float32')
		return appliance_powers
