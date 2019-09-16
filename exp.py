
from api import API

redd = {
  'power': {
    'mains': ['apparent','active'],
    'appliance': ['apparent','active']
  },
  'sample_rate': 60,
    'appliances': ['fridge','air conditioner','electric furnace','washing machine'],
  'methods': {
     'Mean': {},"FHMM_EXACT":{},"RNN":{}
  },
   'train': {    
    'datasets': {
            'Dataport': {
                'path': 'dataport.hdf5',
				'buildings': {
				10: {
					'start_time': '2015-04-04',
					'end_time': '2015-04-24'
				},
				15: {
					'start_time': '2015-04-30',
					'end_time': '2015-05-20'
				},
				37: {
					'start_time': '2014-08-22',
					'end_time': '2014-09-12'
				}
				}
				                
			}
			}
	},
	'test': {
	'datasets': {
		'Datport': {
			'path': 'dataport.hdf5',
			'buildings': {
				10: {
					'start_time': '2015-04-24',
					'end_time': '2015-05-01'
					},
				15: {
					'start_time': '2015-05-20',
					'end_time': '2015-05-27'
					},
				37: {
					'start_time': '2014-08-22',
					'end_time': '2014-08-29'
					}
			}
	}
},
        'metrics':['mae']
}
}


import numpy as np
import pandas as pd

api_res = API(redd)

vals = np.concatenate([np.expand_dims(df.values,axis=2) for df in api_res.errors],axis=2)


cols = api_res.errors[0].columns
indexes = api_res.errors[0].index


mean = np.mean(vals,axis=2)
var = np.var(vals,axis=2)
print ('\n\n')
print (pd.DataFrame(mean,index=indexes,columns=cols))
print ('\n\n')
print (pd.DataFrame(var,index=indexes,columns=cols))