import cvxpy as cvx
import numpy as np
import pandas as pd

from nilmtk_contrib.disaggregate import AFHMM


class AFHMM_SAC(AFHMM):
    """
    Additive Factorial Hidden Markov Model with Signal Aggregate Constraints
    See: http://papers.nips.cc/paper/5526-signal-aggregate-constraints-in-additive-factorial-hmms-with-application-to-energy-disaggregation.pdf
    """
    def __init__(self, params):
        super().__init__(params)
        self.MODEL_NAME = 'AFHMM_SAC'

    def disaggregate_thread(self, test_mains,index,d):
        means_vector = self.means_vector
        pi_s_vector = self.pi_s_vector
        means_vector = self.means_vector
        transmat_vector = self.transmat_vector
        sigma = 100*np.ones((len(test_mains),1))
        flag = 0
        for epoch in range(6):
            if epoch%2==1:
                # The alernative Minimization
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
                            state_vector = cvx.Variable(
                                    shape=(len(test_mains),
                                    self.default_num_states),
                                    name='state_vec-%s'%(appliance_id)
                            )
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
                                matrix = cvx.Variable(
                                        shape=(self.default_num_states, self.default_num_states),
                                        name='variable_matrix-%s-%d'%(appliance_id,t)
                                )
                                appliance_variable_matrix.append(matrix)

                            cvx_variable_matrices.append(appliance_variable_matrix)
                            # Enforcing that their values are ranged
                            for t in range(len(test_mains)):                
                                constraints+=[cvx_variable_matrices[appliance_id][t]>=0]
                                constraints+=[cvx_variable_matrices[appliance_id][t]<=1]

                            # Constraint 6e
                            for t in range(0,len(test_mains)): # 6e
                                for i in range(self.default_num_states):
                                    constraints += [
                                            cvx.sum(((cvx_variable_matrices[appliance_id][t]).T)[i]) == cvx_state_vectors[appliance_id][t][i]
                                    ]

                            # Constraint 6d
                            for t in range(1,len(test_mains)): # 6d
                                for i in range(self.default_num_states):
                                    constraints += [
                                            cvx.sum(cvx_variable_matrices[appliance_id][t][i]) == cvx_state_vectors[appliance_id][t-1][i]
                                    ]

                    for appliance_id in range(self.num_appliances):
                        appliance_usage = cvx_state_vectors[appliance_id]@means_vector[appliance_id]
                        total_appliance_usage = cvx.sum(appliance_usage)
                        constraints += [
                                total_appliance_usage <= self.signal_aggregates[self.appliances[appliance_id]]
                        ]

                    # Second order cone constraints
                    total_observed_reading = np.zeros((test_mains.shape))
                    for appliance_id in range(self.num_appliances):
                        total_observed_reading += cvx_state_vectors[appliance_id]@means_vector[appliance_id]
                    flag=1
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
                prob = cvx.Problem(expression, constraints)
                prob.solve(solver=cvx.SCS,verbose=False, warm_start=True)
                s_ = [
                    np.zeros((len(test_mains), self.default_num_states)) if i.value is None
                    else i.value
                    for i in cvx_state_vectors
                ]

        prediction_dict = {}
        for appliance_id in range(self.num_appliances):
            app_name = self.appliances[appliance_id]
            app_usage= np.sum(s_[appliance_id]@means_vector[appliance_id],axis=1)
            prediction_dict[app_name] = app_usage.flatten()

        d[index] =  pd.DataFrame(prediction_dict,dtype='float32')

