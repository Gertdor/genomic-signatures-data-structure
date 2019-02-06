from operator import itemgetter

class VPTreeNode:
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

class VPTree:

    def createVPTree(values):
        if(len(values) == 0):
            return(None)
        if(len(values)==1):
            return(VPTreeNode(values,None,None))

        currentNodeValue = values[0]
        distances = [(currentNodeValue.distance(x), x) for x in values[1:]]
        distances.sort(key=itemgetter(0))
        median = len(distances)//2
        leftValues = [x[1] for x in distances[:median]]
        rightValues = [x[1] for x in distances[median:]]
        left = VPTree.createVPTree(leftValues)
        right = VPTree.createVPTree(rightValues)
        return(VPTreeNode(currentNodeValue, left, right))
