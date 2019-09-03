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
class AFHMM(Disaggregator):

    def __init__(self, params):
        self.model = []
        
        self.MODEL_NAME = 'AFHMM'  # Add the name for the algorithm
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        if self.load_model_path:
            self.load_model(self.load_model_path)

        self.default_num_states = 2
        self.models = []
        self.num_appliances = 0
        self.time_period = 1440
        self.appliances = []

    def partial_fit(self, train_main, train_appliances, **load_kwargs):

        train_main = pd.concat(train_main, axis=0)
        train_app_tmp = []

        for app_name, df_list in train_appliances:
            df_list = pd.concat(df_list, axis=0)
            train_app_tmp.append((app_name,df_list))
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

        self.means_vector = means_vector
        self.pi_s_vector = pi_s_vector
        self.means_vector = means_vector
        self.transmat_vector = transmat_vector
        print (means_vector)
        print ("Finished Training")

    def disaggregate_chunk(self, test_mains_list):


        means_vector = self.means_vector
        pi_s_vector = self.pi_s_vector
        means_vector = self.means_vector
        transmat_vector = self.transmat_vector

        test_mains_list = pd.concat(test_mains_list,axis=0)
        expression = 0
        sigma = np.ones((len(test_mains_list)))   # The initial vector of Sigmas      
        
        test_mains_big = test_mains_list.values.flatten().reshape((-1,1))
        #print (len(test_mains))
        arr_of_results = []
        for test_block in range(int(math.ceil(len(test_mains_big)/self.time_period))):
            test_mains = test_mains_big[test_block*(self.time_period):(test_block+1)*self.time_period]

            for epoch in range(6):
                if epoch%2==1:
                    # The alernative Minimization
                    usage = np.zeros((len(test_mains)))
                    for appliance_id in range(self.num_appliances):
                        s_v = s_[appliance_id]
                        s_v = np.where(s_v>1,1,s_v)
                        s_v = np.where(s_v<0,0,s_v)
                        app_usage= np.sum(s_v@means_vector[appliance_id],axis=1)
                        usage+=app_usage 
                    sigma = test_mains.flatten() - usage.flatten()

                if epoch%2==0:
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
                                constraints+=[cvx.sum(cvx_variable_matrices[appliance_id][t][i]) == cvx_state_vectors[appliance_id][t][i]]
                        # Constraint 6d
                        for t in range(1,len(test_mains)): # 6d
                            for i in range(self.default_num_states):
                                constraints+=[cvx.sum((cvx_variable_matrices[appliance_id][t]).T[i]) == cvx_state_vectors[appliance_id][t-1][i]]
                    # Second order cone constraints
                    soc_constraints = []

                    total_observed_reading = np.zeros((test_mains.shape))
                        
                        #print (len(cvx_state_vectors))
                    for appliance_id in range(self.num_appliances):
                                total_observed_reading+=cvx_state_vectors[appliance_id]@means_vector[appliance_id]

                    for t in range(len(test_mains)):
                                    soc_constraints+=[cvx.SOC(delta[t], test_mains[t] -  total_observed_reading[t])]

                        
                    constraints+=soc_constraints
                    # intializing the Expression
                    
                    expression = 0                    
                    for appliance_id in range(self.num_appliances):
                    
                        # First loop is over appliances

                        variable_matrix = cvx_variable_matrices[appliance_id]
                        
                        transmat = transmat_vector[appliance_id]
                        # Next loop is over different time-stamps
                        
                        for matrix in variable_matrix:
                            expression-=cvx.sum(cvx.multiply(matrix,np.log(transmat)))
                        
                        one_hot_states = cvx_state_vectors[appliance_id]
                        pi = pi_s_vector[appliance_id]

                        # The expression involving start states
                        first_one_hot_states = one_hot_states[0]
                        #print ("Pis")
                        #print (first_one_hot_states.shape)
                        #print (pi.shape)
                        expression-= cvx.sum(cvx.multiply(first_one_hot_states,np.log(pi)))
                            
                    #print (delta.shape)
                    #print (sigma.shape)
                    
                    for t in range(len(test_mains)):
                        #print (delta[i].shape)
                        #print (sigma[i].shape)
                        if sigma[t]>.8:
                            expression+=.5 * (delta[t][0] / (sigma[t]**2))
                        else:
                            expression+=.5 * (delta[t][0])

                    expression = cvx.Minimize(expression)
                    constraints+=[delta>=0]
                    u = time.time()
                    #print (sigma.shape)
                    prob = cvx.Problem(expression, constraints)
                    #prob.solve(solver=cvx.ECOS_BB)
                    prob.solve(cvx.SCS, verbose=False)
                    print (prob.value)
                    print (time.time()-u)
                    s_ = [i.value for i in cvx_state_vectors]
#                     print (delta.value)
#                     print (s_[0])
#                     print (np.sum(s_[0],axis=1))
#                     print (cvx_variable_matrices[0][0].value)
#                     print (cvx_variable_matrices[0][1].value)
                    print ("Over Iteration")
                    print ("\n\n")

            prediction_dict = {}

            for appliance_id in range(self.num_appliances):
                app_name = self.appliances[appliance_id]

                app_usage= np.sum(s_[appliance_id]@means_vector[appliance_id],axis=1)
                prediction_dict[app_name] = app_usage.flatten()
                #usage+=app_usage

            arr_of_results.append(pd.DataFrame(prediction_dict,dtype='float32'))


        return [pd.concat(arr_of_results,axis=0)]
        #    #print (one_hot_states.shape)
        #    #print (means_vector[appliance_id].shape)
        #    #print (term_3.shape)
        #    #print ((np.sum(one_hot_states*means_vector[appliance_id],axis=1).shape))
        #    term_3+= np.sum(one_hot_states*means_vector[appliance_id],axis=1)

            
        #    #print (term_3.shape)
        #    # term_1_list.append(variable_matrix*np.log(transmat))
        #    # term_2_list.append(one_hot_states*np.log(pi))
        #    #term_3+=hone#appliance_power.values.reshape((-1,1))

        # #print (np.array(term_1_list).shape,np.array(term_2_list).shape,term_3.shape)
        # # term_1_list = np.array(term_1_list)
        # # term_2_list = np.array(term_2_list)



        # # sigma = 30

        # # expression = 0
        # # s = 0
        # # for appliance_id in range(len(term_1_list)):

        # #  for t in range(len(term_1_list[appliance_id])):

        # #      matrix =  term_1_list[appliance_id, t]
        # #      s-=np.sum(matrix)
        # #      #print (matrix.shape)
        # #      #print (matrix)
        # #      expression-=cvx.sum(matrix)



        # # for appliance_id in range(len(term_2_list)):
        # #  matrix = term_2_list[appliance_id]
        # #  s-=np.sum(matrix)
        # #  #print (matrix)
        # #  expression-=cvx.sum(matrix)

        # #print (train_main.shape)
        # #print (expression.value)
        # sigma = 30
        # expression+= .5 * (cvx.norm(train_main - term_3) **2)/(sigma**2)

        # constraints = [one_hot_states_vector<=1, one_hot_states_vector>=0, appliance_variable_matrix>=0,appliance_variable_matrix<=1]

        # exp = '['
        # for appliance_id in range(len(one_hot_states_vector)):

        #    states_vector = one_hot_states_vector[appliance_id]

        #    for t in range(len(states_vector)):
        #        exp+='cvx.sum(one_hot_states_vector[%s,%s])==1'%(appliance_id,t)
        #        exp+=','
        # exp = exp[:-1]
        # exp+=']'
        
        # constraints+=eval(exp)

        # print (expression.value)

        # expression = cvx.Minimize(expression)

        # prob = cvx.Problem(expression, constraints)

        # H = cvx.Variable(shape=(10, 20,30), name='H')
        # print (prob.value) 
        # #print (eval(exp))

        # #print (expression.value)

        # # plt.figure(figsize=(20,8))
        # # plt.plot(train_main.flatten())
        # # plt.plot(term_3.flatten())
        # # plt.show()
        # #expresssion = -cvx.sum(term_1_list) - cvx.sum(term_2_list) + 

        # # print (expression.value)
        # # print (s)

        # # constraints = []

        # # s = "["
        # # for appliance_id in range(len(states_vector)):

        # #  for t in range(len(appliance_id)):
        # #      s+=



        # # term_1 = variable_matrix*(transmat.T)
        # # print (one_hot_states.shape)
        # # term_2 = one_hot_states[0]*pi

        # # #term_3 =  train_main - X
        # # print (term_2.shape, term_1.shape,term_3.shape)
        # # print (train_main.shape)
        #    #print (variable_matrix[1])
        #    #print (variable_matrix[-1])
        #    #print (appliance_states)


        # #print (learnt_model[appliance_name].transmat_)
        # # print (means_vector[0])
        # # print (means_vector[1])
        # # print (states)
        # # print (one_hot_targets)
        # # means_vector = []

        # # for appliance_name in learnt_model:

        # #  for mean_val in learnt_model[appliance_name].predict(X):
        # #      means_vector.append(mean_val)
