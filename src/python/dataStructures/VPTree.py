from operator import itemgetter

class VPTreeNode:
    def __init__(self, value, threshold, left, right):
        self.value = value
        self.threshold = threshold
        self.left = left
        self.right = right

class VPTree:
        
    def createVPTree(values):
        if(len(values) == 0):
            return(None)
        if(len(values)==1):
            return(VPTreeNode(values[0],0,None,None))
        # TODO there might be a smarter way to select this element
        currentNodeValue = values[0]
        distances = [(currentNodeValue.distance(x), x) for x in values[1:]]
        distances.sort(key=itemgetter(0))
        median = len(distances)//2
        threshold = currentNodeValue.distance(distances[median][1])
        leftValues = [x[1] for x in distances[:median]]
        rightValues = [x[1] for x in distances[median:]]
        left = VPTree.createVPTree(leftValues)
        right = VPTree.createVPTree(rightValues)
        return(VPTreeNode(currentNodeValue, threshold, left, right))
    
    def printTree(tree):
        raise NotImplementedError("TODO")

    def nearestNeighbour(tree, point):
        bestDist = tree.value.distance(point)
        return(VPTree.NNS(tree, point, tree, bestDist, 0))

    def NNS(currentNode, point, bestNode, bestDist, ops):
        if(currentNode == None):
            return(bestNode, bestDist, ops)
        ops = ops + 1
        distance = currentNode.value.distance(point)
        if(distance < bestDist):
            bestDist = distance
            bestNode = currentNode

        # Might be faster without this
        if(currentNode.left == None and currentNode.right == None):
            return(bestNode, bestDist, ops)
       
        if(distance < currentNode.threshold):
            if(distance- bestDist < currentNode.threshold):
                (bestNode, bestDist, ops) = VPTree.NNS(currentNode.left, point, bestNode, bestDist, ops)
            if(distance + bestDist > currentNode.threshold):
                (bestNode, bestDist, ops) = VPTree.NNS(currentNode.right, point, bestNode, bestDist, ops)
        else:
            if(distance + bestDist > currentNode.threshold):
                (bestNode, bestDist, ops) = VPTree.NNS(currentNode.right, point, bestNode, bestDist, ops)
            if(distance - bestDist < currentNode.threshold):
               (bestNode, bestDist, ops) = VPTree.NNS(currentNode.left, point, bestNode, bestDist, ops)

        return(bestNode, bestDist, ops)
