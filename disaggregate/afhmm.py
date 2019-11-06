from __future__ import print_function, division
from warnings import warn

import pandas as pd
import numpy as np

from nilmtk.disaggregate import Disaggregator
from hmmlearn import hmm
from collections import OrderedDict

import cvxpy as cvx
from collections import Counter
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
from multiprocessing import Process, Manager
class AFHMM(Disaggregator):

    def __init__(self, params):
        self.model = []
        self.MODEL_NAME = 'AFHMM'        
        self.models = []
        self.num_appliances = 0
        self.appliances = []        
        self.signal_aggregates = OrderedDict()
        self.time_period = 720
        self.time_period = params.get('time_period', self.time_period)
        self.default_num_states = params.get('default_num_states',2)
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.chunk_wise_training =  False
        if self.load_model_path:
            self.load_model(self.load_model_path)


    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        
        self.models = []
        self.num_appliances = 0
        self.appliances = []
        train_main = pd.concat(train_main, axis=0)
        train_app_tmp = []
        for app_name, df_list in train_appliances:
            df_list = pd.concat(df_list, axis=0)
            train_app_tmp.append((app_name,df_list))

        # All the initializations required by the model
        train_appliances = train_app_tmp
        learnt_model = OrderedDict()
        means_vector = []
        one_hot_states_vector = []
        pi_s_vector = []
        transmat_vector = []
        states_vector = []
        train_main = train_main.values.flatten().reshape((-1,1))

        for appliance_name, power in train_appliances:
            #print (appliance_name)
            # Learning the pi's and transistion probabliites  for each appliance using a simple HMM
            self.appliances.append(appliance_name)    
            X = power.values.reshape((-1,1))
            learnt_model[appliance_name] = hmm.GaussianHMM(self.default_num_states, "full")
            # Fit
            learnt_model[appliance_name].fit(X)
            means = learnt_model[appliance_name].means_.flatten().reshape((-1,1))
            states = learnt_model[appliance_name].predict(X)
            transmat = learnt_model[appliance_name].transmat_
            counter = Counter(states.flatten())
            total = 0
            keys = list(counter.keys())
            keys.sort()

            for i in keys:
                total+=counter[i]
            pi = []

            for i in keys:
                pi.append(counter[i]/total)
            pi = np.array(pi)
            nb_classes = self.default_num_states
            targets = states.reshape(-1)
            means_vector.append(means)
            pi_s_vector.append(pi)
            transmat_vector.append(transmat.T)
            states_vector.append(states)
            self.num_appliances+=1
            self.signal_aggregates[appliance_name] = (np.mean(X)*self.time_period).reshape((-1,))

        self.means_vector = means_vector
        self.pi_s_vector = pi_s_vector
        self.means_vector = means_vector
        self.transmat_vector = transmat_vector
        print ("Finished Training")

    def disaggregate_thread(self, test_mains,index,d):

        # A threads that does disaggregation

        means_vector = self.means_vector
        pi_s_vector = self.pi_s_vector
        means_vector = self.means_vector
        transmat_vector = self.transmat_vector

        sigma = 100*np.ones((len(test_mains),1))
        flag = 0

        for epoch in range(6):
            # The alernative Minimization
            if epoch%2==1:
                usage = np.zeros((len(test_mains)))
                for appliance_id in range(self.num_appliances):
                    app_usage= np.sum(s_[appliance_id]@means_vector[appliance_id],axis=1)
                    usage+=app_usage 
                sigma = (test_mains.flatten() - usage.flatten()).reshape((-1,1))
                sigma = np.where(sigma<1,1,sigma)
            else:

                if flag==0:
                    constraints = []
                    cvx_state_vectors = []
                    cvx_variable_matrices = []
                    delta = cvx.Variable(shape=(len(test_mains),1), name='delta_t')
                    for appliance_id in range(self.num_appliances):
                            state_vector = cvx.Variable(shape=(len(test_mains), self.default_num_states), name='state_vec-%s'%(appliance_id))                    
                            cvx_state_vectors.append(state_vector)
                            # Enforcing that their values are ranged
                            constraints+=[cvx_state_vectors[appliance_id]>=0]
                            constraints+=[cvx_state_vectors[appliance_id]<=1]
                            # Enforcing that sum of states equals 1
                            for t in range(len(test_mains)): # 6c
                                constraints+=[cvx.sum(cvx_state_vectors[appliance_id][t])==1]
                            # Creating Variable matrices for every appliance
                            appliance_variable_matrix = []
                            for t in range(len(test_mains)):
                                matrix = cvx.Variable(shape=(self.default_num_states, self.default_num_states), name='variable_matrix-%s-%d'%(appliance_id,t))
                                appliance_variable_matrix.append(matrix)
                            cvx_variable_matrices.append(appliance_variable_matrix)
                            # Enforcing that their values are ranged
                            for t in range(len(test_mains)):                
                                constraints+=[cvx_variable_matrices[appliance_id][t]>=0]
                                constraints+=[cvx_variable_matrices[appliance_id][t]<=1]
                            # Constraint 6e
                            for t in range(0,len(test_mains)): # 6e
                                for i in range(self.default_num_states):
                                    constraints+=[cvx.sum(((cvx_variable_matrices[appliance_id][t]).T)[i]) == cvx_state_vectors[appliance_id][t][i]]
                            # Constraint 6d
                            for t in range(1,len(test_mains)): # 6d
                                for i in range(self.default_num_states):
                                    constraints+=[cvx.sum(cvx_variable_matrices[appliance_id][t][i]) == cvx_state_vectors[appliance_id][t-1][i]]

                    
                    
                    total_observed_reading = np.zeros((test_mains.shape))
                    # TOtal observed reading equals the sum of each appliance
                    for appliance_id in range(self.num_appliances):
                                total_observed_reading+=cvx_state_vectors[appliance_id]@means_vector[appliance_id]                    

                    # Loss function to be minimized
                    
                    term_1 = 0
                    term_2 = 0
                    for appliance_id in range(self.num_appliances):
                        # First loop is over appliances
                        variable_matrix = cvx_variable_matrices[appliance_id]
                        transmat = transmat_vector[appliance_id]
                        # Next loop is over different time-stamps
                        for matrix in variable_matrix:
                            term_1-=cvx.sum(cvx.multiply(matrix,np.log(transmat)))
                        one_hot_states = cvx_state_vectors[appliance_id]
                        pi = pi_s_vector[appliance_id]
                        # The expression involving start states
                        first_one_hot_states = one_hot_states[0]
                        term_2-= cvx.sum(cvx.multiply(first_one_hot_states,np.log(pi)))
                    flag = 1

                expression = 0
                term_3 = 0
                term_4 = 0


                for t in range(len(test_mains)):
                        term_4+= .5 * ((test_mains[t][0] - total_observed_reading[t][0])**2 / (sigma[t]**2))      
                        term_3+= .5 * (np.log(sigma[t]**2))
                expression = term_1 + term_2 + term_3 + term_4
                expression = cvx.Minimize(expression)
                u = time.time()                
                prob = cvx.Problem(expression, constraints,)                
                prob.solve(solver=cvx.SCS,verbose=False,warm_start=True)
                s_ = [i.value for i in cvx_state_vectors]

        prediction_dict = {}
        for appliance_id in range(self.num_appliances):
            app_name = self.appliances[appliance_id]
            app_usage= np.sum(s_[appliance_id]@means_vector[appliance_id],axis=1)
            prediction_dict[app_name] = app_usage.flatten()

        # Store the result in the index corresponding to the thread.

        d[index] =  pd.DataFrame(prediction_dict,dtype='float32')

    def disaggregate_chunk(self, test_mains_list):

        # Sistributes the test mains across multiple threads and runs them in parallel
        manager = Manager()
        d = manager.dict()
        
        predictions_lst = []
        for test_mains in test_mains_list:        
            test_mains_big = test_mains.values.flatten().reshape((-1,1))
            self.arr_of_results = []        
            st = time.time()
            threads = []
            for test_block in range(int(math.ceil(len(test_mains_big)/self.time_period))):
                test_mains = test_mains_big[test_block*(self.time_period):(test_block+1)*self.time_period]
                t = Process(target=self.disaggregate_thread, args=(test_mains,test_block,d))
                threads.append(t)

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            for i in range(len(threads)):
                self.arr_of_results.append(d[i])
            prediction = pd.concat(self.arr_of_results,axis=0)
            predictions_lst.append(prediction)
            
        return predictions_lst



