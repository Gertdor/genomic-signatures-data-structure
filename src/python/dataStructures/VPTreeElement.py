import numpy as np

class VPTreeElement:

    def __init__(self, value):
        self.value = value

    def distance(self, other):
        return(np.linalg.norm(self.value-other.value))
