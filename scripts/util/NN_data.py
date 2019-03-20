class NNData:
    """ Stores data from multiple NN runs
    for example from running the greedy test in gsTest.py

    """

    # saker att lagra
    # k
    # trees, searched points
    # all runs, [] - each greedy factor
    # for each greedy factor, [obs]

    def __init__(self, all_runs, greedy_factors, all_signatures_used):
        self.greedy_factors = greedy_factors
        self.signatures_used = all_signatures_used
        self.all_runs = all_runs
        self.k = len(all_runs[0][0][0])

    def get_distances_by_factor(self):

        return [
            [nn[0] for tree in run for NNS in run for knn in NNS for nn in knn[0]]
            for run in self.all_runs
        ]

    def get_ids_by_factor(self):
        return [
            [nn[1].identifier for tree in run for NNS in run for knn in NNS for nn in knn[0]]
            for run in self.all_runs
        ]

    def get_ops_by_factor(self, repeat_k=False):
        if repeat_k:
            return [
                [
                    knn[1]
                    for tree in run
                    for NNS in run
                    for knn in NNS
                    for i in range(self.k)
                ]
                for run in self.all_runs
            ]
        else:
            return [
                [knn[1] for tree in run for NNS in run for knn in NNS]
                for run in self.all_runs
            ]

    def get_signatures_used(self):
        return self.signatures_used

    def get_greedy_factors(self):
        return self.greedy_factors

    def get_node_ids(self):
        return 0
