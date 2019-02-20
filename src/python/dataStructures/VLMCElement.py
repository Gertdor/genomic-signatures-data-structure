from clustering_genomic_signatures.distance import FrobeniusNorm

class VPTreeVLMC:

    def __init__(self, value):
        self.value = value
        self.dist_func = FrobeniusNorm()
    
    def distance(self, other):
        return(self.dist_func.distance(self.value, other.value))
