from operator import itemgetter
from sys import float_info
from random import randint

import heapq
import pickle

class VPTreeNode:
    
    def __init__(self, value, threshold, left, right):
        self.value = value
        self.threshold = threshold
        self.left = left
        self.right = right
    
    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if(self.value!=other.value):
            return False
        if(self.threshold!=other.threshold):
            return False
        return(self.left == other.left and self.right == other.right)

    def __ne__(self, other):
        return not(self==other)
    

class VPTree:
        
    def save(tree, fileName):
        with open(fileName, "wb") as f:
            pickle.dump(tree,f)

    def load(fileName):
        with open(fileName, "rb") as f:
            return pickle.load(f)

    def createVPTree(values):
        if(len(values) == 0):
            return(None)
        if(len(values)==1):
            return(VPTreeNode(values[0],0,None,None))
        # TODO there might be a smarter way to select this element
        index = randint(0,len(values)-1)
        currentNodeValue = values[index]
        tmp = values[0]
        values[0] = values[index]
        values[index] = tmp
        distances = [(currentNodeValue.distance(x), x) for x in values[1:]]
        distances.sort(key=itemgetter(0))
        median = len(distances)//2
        threshold = currentNodeValue.distance(distances[median][1])
        leftValues = [x[1] for x in distances[:median]]
        rightValues = [x[1] for x in distances[median:]]
        left = VPTree.createVPTree(leftValues)
        right = VPTree.createVPTree(rightValues)
        return(VPTreeNode(currentNodeValue, threshold, left, right))
    
    def toJson(tree, level = 0):
        if(tree is None):
            print("None", end='')
        else:
            indent = "  "*level
            print("{")
            print(indent,"value:",tree.value.value,',', sep='')
            print(indent,"threshold:",tree.threshold,',', sep='')
            print(indent,"left:", end='',sep='')
            VPTree.toJson(tree.left, level + 1)
            print(",")
            print(indent,"right:", end='',sep='')
            VPTree.toJson(tree.right, level + 1)
            print()
            print(indent,"}", end='',sep='')

    def nearestNeighbour(tree, point, k = 1):
        dist = tree.value.distance(point)
        if(k==1):
            cutOffDist = dist
            bestNodes = [(dist,tree)]
        else:
            bestNodes = [(float_info.max, None) for i in range(k-1)]
            bestNodes.append((dist, tree))
            cutOffDist = float_info.max
        return(VPTree.NNS(tree, point, bestNodes, cutOffDist, 0))

    def NNS(currentNode, point, bestNodes, cutOffDist, ops):
        if(currentNode is None):
            return(bestNodes, cutOffDist, ops)
        ops = ops + 1
        distance = currentNode.value.distance(point)
        if(distance < cutOffDist):
            (maxDist,currentIndex) = VPTree.knnMax(bestNodes)
            bestNodes[currentIndex] = (distance, currentNode)
            (cutOffDist, currentIndex) = VPTree.knnMax(bestNodes)

        # Might be faster without this
        if(currentNode.left is None and currentNode.right is None):
            return(bestNodes, cutOffDist, ops)
       
        if(distance < currentNode.threshold):
            (bestNodess, cutOffDist, ops) = VPTree.NNS(currentNode.left, point, bestNodes, cutOffDist, ops)
            if(distance + cutOffDist > currentNode.threshold):
                (bestNodess, cutOffDist, ops) = VPTree.NNS(currentNode.right, point, bestNodes, cutOffDist, ops)
        else:
            (bestNodes, cutOffDist, ops) = VPTree.NNS(currentNode.right, point, bestNodes, cutOffDist, ops)
            if(distance - cutOffDist < currentNode.threshold):
               (bestNodes, cutOffDist, ops) = VPTree.NNS(currentNode.left, point, bestNodes, cutOffDist, ops)

        return(bestNodes, cutOffDist, ops)


    def knnMax(nodeList):
        length = len(nodeList)
        if(len==1):
            return(nodeList[0][0],0)
        maxDist = 0
        currentIndex = 0
        for i in range(length):
            if(maxDist < nodeList[i][0]):
                maxDist = nodeList[i][0]
                currentIndex = i
        return(maxDist, currentIndex)
