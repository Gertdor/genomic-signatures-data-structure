from dataStructures.VPTree import VPTree
import numpy as np
from random import randint, seed
from dataStructures.VPTreeElement import VPTreeElement

import cProfile

k = 1
dim = 10
numberList = [VPTreeElement(np.random.uniform(0, 3, dim)) for x in range(70000)]
tree = VPTree.createVPTree(numberList)


numberOfElements = 100
elements = [
    VPTreeElement(np.random.uniform(0, 3, dim)) for x in range(numberOfElements)
]

NNS = [VPTree.nearestNeighbour(tree, elem, k) for elem in elements]
# cProfile.run('VPTree.nearestNeighbour(tree,elementToCompare,k)')

# distances = [(elementToCompare.distance(x), x) for x in numberList]
# distances.sort(key=lambda x: x[0])
# NN.sort(key=lambda x:x[0])


# print("number of ops", ops)
total_distance_calls = 0
for NN in NNS:
    # for i in range(k):
    # dist = NN[i][0]
    # element = NN[i][1]
    dist = NN[0][0][0]
    element = NN[0][0][1]
    total_distance_calls += NN[2]
    # print("Best distances: ",round(distances[i][0],3))
    # print("Best element: ", distances[i][1].value)
    # print("Found Element: ", element.value.value)
    # print("Found distance: ", round(dist,3))
    # print("\n")

print("avg number of distance calls", total_distance_calls / numberOfElements)
# VPTree.toJson(tree) # Prints tree
