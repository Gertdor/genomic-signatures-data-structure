import numpy as np


class VPTreeElement:
    def __init__(self, value, identifier=None):
        self.value = value
        self.identifier = identifier

    def distance(self, other):
        return np.linalg.norm(self.value - other.value)

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value
