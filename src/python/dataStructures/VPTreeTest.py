from VPTree import VPTree
import numpy as np
from random import randint, seed
from VPTreeElement import VPTreeElement

k=1

numberList = [VPTreeElement(np.array([randint(1,101),randint(1,101),randint(1,101)])) for x in range(10000)]
tree = VPTree.createVPTree(numberList)
elementToCompare = VPTreeElement(np.array([50,2,17]))
(NN, dist, ops) = VPTree.nearestNeighbour(tree,elementToCompare,k)
distances = [(elementToCompare.distance(x), x) for x in numberList]
distances.sort(key=lambda x: x[0])
NN.sort(key=lambda x:x[0])


print("number of ops", ops)
for i in range(k):
    dist = NN[i][0]
    element = NN[i][1]
    print("Best distances: ",round(distances[i][0],3))
    print("Best element: ", distances[i][1].value)
    print("Found Element: ", element.value.value)
    print("Found distance: ", round(dist,3))
    print("\n")

#VPTree.toJson(tree) # Prints tree
