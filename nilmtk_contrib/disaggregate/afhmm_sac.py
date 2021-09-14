import cvxpy as cvx

from nilmtk_contrib.disaggregate import AFHMM


class AFHMM_SAC(AFHMM):
    """
    Additive Factorial Hidden Markov Model with Signal Aggregate Constraints
    See: http://papers.nips.cc/paper/5526-signal-aggregate-constraints-in-additive-factorial-hmms-with-application-to-energy-disaggregation.pdf
    """
    def __init__(self, params):
        super().__init__(params)
        self.MODEL_NAME = 'AFHMM_SAC'

    def setup_cvx_constraints(self, n_samples, n_appliances):
        cvx_state_vectors, constraints, cvx_variable_matrices = super().setup_cvx_constraints(n_samples, n_appliances)
        # Constraints on signal aggregates.
        for i, (appli_name, signal_aggregate) in enumerate(self.signal_aggregates.items()):
            appliance_usage = cvx_state_vectors[i]@self.means_vector[appli_name]
            total_appliance_usage = cvx.sum(appliance_usage)
            constraints.append(total_appliance_usage <= signal_aggregate)

        return cvx_state_vectors, constraints, cvx_variable_matrices

