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

    def setup_cvx_constraints(self, n_samples, n_appliances):
        cvx_state_vectors = [
                cvx.Variable(
                    shape=( n_samples, self.default_num_states ),
                    name="state_vec-{}".format(i)
                )
                for i in range(n_appliances)
        ]
        constraints = []
        for stv in cvx_state_vectors:
            # State vector values are ranged.
            constraints += [ stv >= 0, stv <= 1 ]
            # Sum of states equals 1.
            for t in range(n_samples):
                constraints.append(cvx.sum(stv[t]) == 1)
        # Create variable matrices for each appliance, for each sample.
        cvx_variable_matrices = [
                [
                    cvx.Variable(
                        shape=( self.default_num_states, self.default_num_states ),
                        name="variable_matrix-{}-{}".format(i, t)
                    )
                    for t in range(n_samples)
                ]
                for i in range(n_appliances)
        ]
        for i, appli_varmats in enumerate(cvx_variable_matrices):
            for t, varmat in enumerate(appli_varmats):
                # Assign range constraints to variable matrices.
                constraints += [ varmat >= 0, varmat <= 1 ]
                # Assign equality constraints with state vectors.
                constraints += [
                    cvx.sum(varmat[l]) == cvx_state_vectors[i][t-1][l]
                    for l in range(self.default_num_states)
                ]
                constraints += [
                    cvx.sum((varmat.T)[l]) == cvx_state_vectors[i][t][l]
                    for l in range(self.default_num_states)
                ]

        return cvx_state_vectors, constraints, cvx_variable_matrices

    def disaggregate_thread(self, test_mains):
        n_epochs = 6 # don't put in __init__, those are inference epochs!
        n_samples = len(test_mains)
        sigma = 100*np.ones(( n_samples, 1 ))
        cvx_state_vectors, constraints, cvx_varmats = self.setup_cvx_constraints(
                n_samples, len(self.means_vector))
        # Preparing first terms of the objective function.
        term_1 = 0
        term_2 = 0
        total_appli_energy = np.zeros_like(test_mains)
        for i, (appli_name, means) in enumerate(self.means_vector.items()):
            total_appli_energy += cvx_state_vectors[i]@means
            appli_varmats = cvx_varmats[i]
            transmat = self.transmat_vector[appli_name]
            for varmat in appli_varmats:
                term_1 -= cvx.sum(cvx.multiply(varmat, np.log(transmat)))

            first_hot_state = cvx_state_vectors[i][0]
            transition_p = self.pi_s_vector[appli_name]
            term_2 -= cvx.sum(cvx.multiply(first_hot_state, np.log(transition_p)))

        for epoch in range(n_epochs):
            if epoch % 2:
                # Alernative minimization on odd epochs.
                usage = np.zeros(( n_samples, ))
                for i, (appli_name, means) in enumerate(self.means_vector.items()):
                    usage += np.sum(s_[i]@means, axis=1)
                sigma = (test_mains.flatten() - usage.flatten()).reshape(( -1, 1 ))
                sigma = np.where(sigma < 1, 1, sigma)
            else:
                # Primary minimization on even epochs.
                term_3 = 0
                term_4 = 0
                for t in range(n_samples):
                    term_3 += .5 * (np.log(sigma[t]**2))
                    term_4 += .5 * ((test_mains[t][0] - total_appli_energy[t][0])**2 / (sigma[t]**2))

                objective = cvx.Minimize(term_1 + term_2 + term_3 + term_4)
                prob = cvx.Problem(objective, constraints)
                prob.solve(solver=cvx.SCS, verbose=False, warm_start=True)
                s_ = [
                        np.zeros((n_samples, self.default_num_states)) if i.value is None
                        else i.value
                        for i in cvx_state_vectors
                ]

        prediction_dict = {}
        for i, (appli_name, means) in enumerate(self.means_vector.items()):
            app_usage = np.sum(s_[i]@means, axis=1)
            prediction_dict[appli_name] = app_usage.flatten()

        return pd.DataFrame(prediction_dict, dtype="float32")

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

