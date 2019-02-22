from clustering_genomic_signatures.distance import FrobeniusNorm

class VPTreeVLMC:

    def __init__(self, value, distance_function, identifier = None):
        self.value = value
        self.dist_func = distance_function
        self.identifier = identifier
    
    def distance(self, other):
        return(self.dist_func.distance(self.value, other.value))
