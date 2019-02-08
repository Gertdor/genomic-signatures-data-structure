from VPTree import VPTree
import numpy as np
from random import randint
from VPTreeElement import VPTreeElement

# Should generate a large tree.
# Tests so that i can store any value.
numberList = [VPTreeElement(np.array([randint(1,101),randint(1,101),randint(1,101)])) for x in range(10000)]
a = VPTree.createVPTree(numberList)
(NN, dist, ops) = VPTree.nearestNeighbour(a,VPTreeElement(np.array([50,2,17])))
print(ops)

# Specific tree with known structure
a = VPTree.createVPTree([VPTreeElement(np.array([1,3])),VPTreeElement(np.array([2,5])),VPTreeElement(np.array([3,3]))])
(NN, dist, ops) = VPTree.nearestNeighbour(a,VPTreeElement(np.array([2,3])))
#NN.value.print()
print(dist)

