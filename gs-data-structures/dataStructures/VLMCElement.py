from clustering_genomic_signatures.distance import FrobeniusNorm


class VPTreeVLMC:
    def __init__(self, value, distance_function, identifier=None):
        self.value = value
        self.dist_func = distance_function
        if identifier is None:
            self.identifier = value.name
        else:
            self.identifier = identifier

    def distance(self, other):
        return self.dist_func.distance(self.value, other.value)

    def __eq__(self, other):
        return self.identifier == other.identifier

    def __ne__(self, other):
        return not (self == other)
