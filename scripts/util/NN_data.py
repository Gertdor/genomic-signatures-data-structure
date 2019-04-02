class NNData:
    """ Stores data from multiple NN runs
    for example from running the greedy test in gsTest.py

    """

    def __init__(self, all_runs, all_signatures_used, factors):
        self.greedy_factors = [f[0] for f in factors]
        self.k_values = [f[1] for f in factors]
        self.gc_prune = [f[2] for f in factors]
        self.signatures_used = all_signatures_used
        run_data = [SingleRunData(all_runs[key]) for key in all_runs]
        self.run_data = run_data

    def get_distances_by_factor(self, un_pack=True):

        return [run.get_distances(un_pack) for run in self.run_data]

    def get_ids_by_factor(self, un_pack=True):
        return [run.get_ids(un_pack) for run in self.run_data]

    def get_ops_by_factor(self, repeat_k=False):
        return [run.get_ops(repeat_k) for run in self.run_data]

    def get_signatures_used(self):
        return self.signatures_used

    def get_greedy_factors(self):
        return self.greedy_factors

    def get_k_values(self):
        return self.k_values

    def get_gc_prune_valunes(self):
        return self.gc_prune


class SingleRunData:
    def __init__(self, run_data):

        self.run_data = run_data

    def get_distances(self, unpack):
        distances = [
            NNS.get_distances() for queries in self.run_data for NNS in queries
        ]
        if unpack:
            distances = [d for NNS in distances for d in NNS]
        return distances

    def get_ids(self, unpack):
        ids = [NNS.get_ids() for queries in self.run_data for NNS in queries]
        if unpack:
            ids = [i for NNS in ids for i in NNS]
        return ids

    def get_ops(self, repeat_k):
        if repeat_k:
            ops = [
                NNS.get_ops()
                for queries in self.run_data
                for NNS in queries
                for k in range(NNS.get_size())
            ]
        else:
            ops = [NNS.get_ops() for queries in self.run_data for NNS in queries]
        return ops
