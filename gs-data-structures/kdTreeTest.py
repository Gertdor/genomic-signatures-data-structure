from dataStructures.kdTree import KdTree
from random import randint
from dataStructures.pointElement import Point

import numpy as np

# Should generate a large tree.
# Tests so that i can store any value.

dim = 3
numberList = [Point(np.random.uniform(0,3,dim)) for x in range(70000)]
a = KdTree.createKdTree(numberList,0)

(NN, dist, distCalc) = KdTree.findNearestNeighbour(a,Point(np.random.uniform(0,3,dim)))

print(distCalc)

# Specific tree with known structure
#a = KdTree.createKdTree([Point(np.array([1,3])),Point(np.array([2,5])),Point(np.array([3,3]))],0)
#KdTree.printTree(a)
#(NN, dist) = KdTree.findNearestNeighbour(a,[2,3])
#NN.value.print()
#print(dist)
#
#
