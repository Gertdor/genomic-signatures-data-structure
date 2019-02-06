import numpy as np
from element import Element

class Point(Element):

    def __init__(self, value):
        self.value = value

    def greaterThan(self, other, axis):
        return(self.value[axis]>other[axis])

    def axisDist(self, other, axis):
        return(self.value[axis]-other[axis])

    def distance(self, other):
        return(np.linalg.norm(self.value-other,ord=1))
