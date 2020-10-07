import cvxpy as cvx
import math
import multiprocessing
import nilmtk.docinherit
import numpy as np
import pandas as pd

from collections import Counter, OrderedDict
from hmmlearn import hmm
from nilmtk.disaggregate import Disaggregator


class AFHMM(Disaggregator):
    """
    Additive Factorial Hidden Markov Model (without Signal Aggregate Constraints)
    See: http://papers.nips.cc/paper/5526-signal-aggregate-constraints-in-additive-factorial-hmms-with-application-to-energy-disaggregation.pdf
    """
    def __init__(self, params):
        self.MODEL_NAME = 'AFHMM'
        self.models = []
        self.means_vector = OrderedDict()
        self.pi_s_vector = OrderedDict()
        self.transmat_vector = OrderedDict()
        self.signal_aggregates = OrderedDict()
        self.time_period = params.get("time_period", 720)
        self.default_num_states = params.get("default_num_states", 2)
        self.save_model_path = params.get("save-model-path", None)
        self.load_model_path = params.get("pretrained-model-path", None)
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        if self.load_model_path:
            self.load_model(self.load_model_path)

    @nilmtk.docinherit.doc_inherit
    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        """
            train_main: pd.DataFrame It will contain the mains reading.
            train_appliances: list of tuples [('appliance1', [ df1 ]),('appliance2', [ df2 ]),...]
        """
        for appli_name, df_list in train_appliances:
            # Compute model parameters for this chunk.
            app_df = pd.concat(df_list, axis=0)
            X = app_df.values.reshape(( -1, 1 ))
            learnt_model = hmm.GaussianHMM(self.default_num_states, "full")
            learnt_model.fit(X)
            means = learnt_model.means_.flatten().reshape(( -1, 1 ))
            states = learnt_model.predict(X)
            transmat = learnt_model.transmat_.T
            counter = Counter(states.flatten())
            total = sum(counter.values())
            pi = np.array([ v/total for v in counter.values() ])
            sigagg = (np.mean(X) * self.time_period).reshape(( -1, ))
            # Merge with previous values.
            # Hypothesis 1: chunk size is constant. (mean of means)
            # Hypothesis 2: if the appliance is already registered in
            # self.means_vector, then it is also known in all other dicts.
            if appli_name in self.means_vector:
                self.means_vector[appli_name] = (self.means_vector[appli_name] + means) / 2
                self.pi_s_vector[appli_name] = (self.pi_s_vector[appli_name] + pi) / 2
                self.transmat_vector[appli_name] = (self.transmat_vector[appli_name] + transmat) / 2
                self.signal_aggregates[appli_name] = (self.signal_aggregates[appli_name] + sigagg) / 2
            else:
                self.means_vector[appli_name] = means
                self.pi_s_vector[appli_name] = pi
                self.transmat_vector[appli_name] = transmat
                self.signal_aggregates[appli_name] = sigagg

        print ("{}: Finished training".format(self.MODEL_NAME))

    def disaggregate_thread(self, test_mains,index,d):
        means_vector = self.means_vector
        pi_s_vector = self.pi_s_vector
        means_vector = self.means_vector
        transmat_vector = self.transmat_vector

        sigma = 100*np.ones((len(test_mains),1))
        flag = 0

        for epoch in range(6):
            # The alernative minimization
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
                            state_vector = cvx.Variable(
                                    shape=(len(test_mains), self.default_num_states), 
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
                                    constraints+=[cvx.sum(((cvx_variable_matrices[appliance_id][t]).T)[i]) == cvx_state_vectors[appliance_id][t][i]]
                            # Constraint 6d
                            for t in range(1,len(test_mains)): # 6d
                                for i in range(self.default_num_states):
                                    constraints+=[
                                            cvx.sum(cvx_variable_matrices[appliance_id][t][i]) == cvx_state_vectors[appliance_id][t-1][i]
                                    ]

                    total_observed_reading = np.zeros((test_mains.shape))
                    # Total observed reading equals the sum of each appliance
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
                prob = cvx.Problem(expression, constraints,)
                prob.solve(solver=cvx.SCS,verbose=False,warm_start=True)
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

        # Store the result in the index corresponding to the thread.
        d[index] =  pd.DataFrame(prediction_dict,dtype='float32')

    @nilmtk.docinherit.doc_inherit
    def disaggregate_chunk(self, test_mains):
        # Distributes the test mains across multiple threads and disaggregate in parallel
        # Use all available CPUs except one for the OS.
        n_workers = max(( 1, multiprocessing.cpu_count() - 1 ))
        predictions_lst = []
        with multiprocessing.Pool(n_workers) as workers:
            for mains_df in test_mains:
                mains_vect = mains_df.values.flatten().reshape(( -1, 1 ))
                n_blocks = int(math.ceil(len(mains_vect)/self.time_period))
                blocks = [
                        mains_vect[b * self.time_period:(b + 1) * self.time_period]
                        for b in range(n_blocks)
                ]
                res_arr = workers.map(self.disaggregate_thread, blocks)
                predictions_lst.append(pd.concat(res_arr, axis=0))

        return predictions_lst

