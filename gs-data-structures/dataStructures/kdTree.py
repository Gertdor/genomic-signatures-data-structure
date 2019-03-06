from operator import itemgetter

# Create a class for the tree and one for the nodes
# Let the tree class contain meta data and a pointer to the root

class KdNode:

    def __init__(self, value, depth, axis, leftChild, rightChild):
        self.value = value
        self.depth = depth
        self.axis = axis
        self.leftChild = leftChild
        self.rightChild = rightChild

class KdTree:

    def createKdTree(values, depth = 0):
        if(len(values) == 0):
            return(None)

        dimOfData = len(values[0].value) # Dimension of data
        axis = depth % dimOfData

        if(len(values) == 1):
            return(KdNode(values[0],depth,axis, None,None))

#        values.sort(key=itemgetter(axis)) #TODO make general
        values.sort(key=lambda x: x.value[axis]) #TODO make general
        
        median = len(values)//2;
        value = values[median]
        left = KdTree.createKdTree(values[:median], depth+1)
        right = KdTree.createKdTree(values[(median+1):], depth+1) #median + 1 because I store a value in the node
        return(KdNode(value, depth, axis, left, right))

    def printTree(tree):
        if(tree!=None):
            KdTree.printTree(tree.leftChild)
            print(tree.value.print())
            KdTree.printTree(tree.rightChild)

    def findNearestNeighbour(tree, point):
        currentNode = tree
        bestDist = tree.value.distance(point)
        bestNode = tree
        return(KdTree.NNS(tree, point, bestNode, bestDist, 0))

    def NNS(currentNode, point, bestNode, bestDist, distCalc):
        if(currentNode == None):
            return(bestNode, bestDist, distCalc)
        distance = currentNode.value.distance(point)
        distCalc+=1
        axisDistance = currentNode.value.axisDist(
                point, currentNode.axis)
        if(distance < bestDist):
            bestDist = distance
            bestNode = currentNode
        
        # determine which subtree(s) to explore, and in what order
        if(currentNode.value.greaterThan(point, currentNode.axis)):
            near = currentNode.leftChild
            far = currentNode.rightChild
        else:
            near = currentNode.rightChild
            far = currentNode.leftChild
        #It cannot get worse, so this is safe.
        (bestNode, bestDist, distCalc) = KdTree.NNS(near, point, bestNode, bestDist, distCalc)
        if(axisDistance*axisDistance>=bestDist*bestDist):
            return(bestNode, bestDist, distCalc)
        return(KdTree.NNS(near, point, bestNode, bestDist, distCalc))
