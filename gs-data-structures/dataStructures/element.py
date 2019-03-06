class Element:

    def __init__(self, value):
        self.value = value

    def print(self):
        print(self.value)

    def greaterThan(self, other, dim):
        raise NotImplementedError("Implement in sub class")

    def axisDist(self, other, axis):
        raise NotImplementedError("Implement in sub class")

    def distance(self, other):
        raise NotImplementedError("Implement in sub class")
