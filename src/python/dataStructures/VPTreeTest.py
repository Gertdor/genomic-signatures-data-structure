from VPTree import VPTree
import numpy as np
from random import randint
from VPTreeElement import VPTreeElement

# Should generate a large tree.
# Tests so that i can store any value.
numberList = [VPTreeElement(np.array([randint(1,11),randint(1,11)])) for x in range(25)]
a = VPTree.createVPTree(numberList)

# Specific tree with known structure
#a = KdTree.createKdTree([Point(np.array([1,3])),Point(np.array([2,5])),Point(np.array([3,3]))],0)
#KdTree.printTree(a)
#(NN, dist) = KdTree.findNearestNeighbour(a,[2,3])
#NN.value.print()
#print(dist)
#a = VPTree.createVPTree(np.array(
