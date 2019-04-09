from collections import Counter


class NNData:
    """ Stores data from multiple NN runs
    for example from running the greedy test in gsTest.py

    """

    def __init__(self, all_runs, all_signatures_used, factors, meta_data=None):
        self.forest_usage = [f[0] for f in factors]
        self.greedy_factors = [f[1] for f in factors]
        self.k_values = [f[2] for f in factors]
        self.gc_prune = [f[3] for f in factors]
        self.signatures_used = all_signatures_used
        run_data = [SingleRunData(all_runs[key]) for key in all_runs]
        self.run_data = run_data
        self.meta_data = meta_data

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

    def classify(self, rank):
        all_names = [run.get_names(False) for run in self.run_data]
        if self.meta_data is None:
            return NNS

        return [
            [self._classify_one(knn, self.meta_data, rank) for knn in NNS]
            for NNS in all_names
        ]

    def _classify_one(self, knn, meta_data, rank):
        taxonomic_data = [meta_data[nn][rank] for nn in knn]
        return Counter(taxonomic_data).most_common(1)[0]

    def get_k_values(self):
        return self.k_values

    def get_gc_prune_valunes(self):
        return self.gc_prune

    def get_keys(self):

        return [
            "Forest" * f + " P: " + str(p) + " K: " + str(k) + " gc" * gc
            for f, p, k, gc in zip(
                self.forest_usage, self.greedy_factors, self.k_values, self.gc_prune
            )
        ]


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

    def get_names(self, unpack):
        names = [NNS.get_names() for queries in self.run_data for NNS in queries]
        if unpack:
            names = [name for NNS in names for name in NNS]
        return names

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
