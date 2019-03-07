from operator import itemgetter
from sys import float_info
from random import randint
import numpy as np

import heapq
import pickle

class VPTreeNode:
    
    def __init__(self, vp, threshold, left, right, data=None):
        self.vp = vp
        self.threshold = threshold
        self.is_leaf = left is None and right is None
        self.left = left
        self.right = right
        self.data = data
    
    def distance(self, other):
        return self.vp.distance(other)

    def __str__(self):
        if(self.data is not None):
            return str(self.vp) + str(self.data)
        else:
            return str(self.vp)

    def __eq__(self, other):
        if(self.vp!=other.vp):
            return False
        if(self.threshold!=other.threshold):
            return False
        if(self.data != other.data):
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
    
    def createVPTree(values, random=True, max_leaf_size = 1):
        if(len(values) == 0):
            return(None)
        if(len(values)<=max_leaf_size):
            # calc distance from every node to every node
            # select one which minimizes threshold
            if(max_leaf_size == 1):
                return(VPTreeNode(values[0], 0, None, None, values))

            distances = [[current.distance(other) for current in values] for other in values]
            worse_case_distances = np.max(distances, axis=0)
            best_index = np.argmin(worse_case_distances)
            threshold = worse_case_distances[best_index]

            values_to_save = values.copy()
            values_to_save[0], values_to_save[best_index] = values_to_save[best_index], values_to_save[0]
            return(VPTreeNode(values_to_save[0],threshold,None,None, values_to_save))

        # TODO there might be a smarter way to select this element
        # Selects a random element and moves it to the front
        # That way it can be used as a pivot
        if random:
            index = randint(0,len(values)-1)
            values[0], values[index] =  values[index], values[0]

        currentNodeValue = values[0]
        distances = [(currentNodeValue.distance(x), x) for x in values[1:]]
        distances.sort(key=itemgetter(0))
        median = len(distances)//2
        threshold = currentNodeValue.distance(distances[median][1])
        leftValues = [x[1] for x in distances[:median]]
        rightValues = [x[1] for x in distances[median:]]
        left = VPTree.createVPTree(leftValues,random,max_leaf_size)
        right = VPTree.createVPTree(rightValues,random,max_leaf_size)
        # swap back to not change the original data
        if random:
            values[0], values[index] =  values[index], values[0]
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
        dist = tree.distance(point)
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
        distance = currentNode.distance(point)
        if(distance < cutOffDist):
            VPTree.insertSorted(bestNodes,(distance,currentNode.vp))
            cutOffDist = bestNodes[0][0]

        if(currentNode.is_leaf):
            if(len(currentNode.data) == 1):
                return(bestNodes, cutOffDist, ops)
            return VPTree.closestLinearSearch(currentNode, point, distance, bestNodes, cutOffDist, ops)

        if(distance < currentNode.threshold):
            (bestNodess, cutOffDist, ops) = VPTree.NNS(currentNode.left, point, bestNodes, cutOffDist, ops)
            if(distance + cutOffDist > currentNode.threshold):
                (bestNodess, cutOffDist, ops) = VPTree.NNS(currentNode.right, point, bestNodes, cutOffDist, ops)
        else:
            (bestNodes, cutOffDist, ops) = VPTree.NNS(currentNode.right, point, bestNodes, cutOffDist, ops)
            if(distance - cutOffDist < currentNode.threshold):
               (bestNodes, cutOffDist, ops) = VPTree.NNS(currentNode.left, point, bestNodes, cutOffDist, ops)

        return(bestNodes, cutOffDist, ops)
    
    def closestLinearSearch(currentNode, point, distance, bestNodes, cutOffDist, ops):

        if(distance - currentNode.threshold < cutOffDist):
            distances = [(vlmc.distance(point),vlmc) for vlmc in currentNode.data]
            ops+=len(currentNode.data)
            distances.sort(key=itemgetter(0))
            for distance in distances:
                if(distance[0] < cutOffDist):
                    VPTree.insertSorted(bestNodes, distance)
                    cutOffDist = bestNodes[0][0]

        return(bestNodes, cutOffDist, ops)
    
    def insertSorted(nodeList, item):
        for i, val in enumerate (nodeList[:-1]):
            if(val[1]>item[1]):
                nodeList[i]=nodeList[i+1]
            else:
                nodeList[i]=item
        nodeList[-1]=item
    
    def overlap(tree):
        node_list=[]
        VPTree.create_list(tree, node_list)
        overlap=0
        for i in range(len(node_list)):
            current=node_list[i]
            overlap+=sum([VPTree._overlap(current, other) for other in node_list[i:]])
        return(overlap)

    def _overlap(current, other):
        return(int(current.distance(other.vp) > (current.threshold + other.threshold)))

    def create_list(tree, node_list):
        if tree is not None:
            node_list.append(tree)
            VPTree.create_list(tree.left, node_list)
            VPTree.create_list(tree.right, node_list)