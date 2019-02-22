import sys
import time
import argparse

import numpy as np
from scipy import stats
sys.path.append('/home/basse/Documents/skola/masterThesis/clustering-genomic-signatures-private')

from clustering_genomic_signatures.util.parse_vlmcs import parse_vlmcs, add_parse_vlmc_args
from clustering_genomic_signatures.util.parse_distance import add_distance_arguments, parse_distance_method
from dataStructures.VPTree import VPTree
from dataStructures.VPTreeElement import VPTreeElement
from dataStructures.VLMCElement import VPTreeVLMC
from data_analysis.distance_analysis import distance_function_stats

def fullTree(elements):
    tree = VPTree.createVPTree(elements)
    NNS = [VPTree.nearestNeighbour(tree, elem) for elem in elements]
    totalTime = start_time - time.time()
    
    print(NNS[0])
    i = 0
    count = 0
    totalNumberOfActions = 0
    for elem in NNS:
        totalNumberOfActions+=elem[2]
        if(elem[0][0][1].value == elements[i]):
            count=count+1
        i=i+1
    
    print(count)
    print(totalNumberOfActions/len(vlmcs))

def partOfTree(cutoff, elements):
    numElemInTree = round(len(elements)*cutoff)
    elementsChecked = len(elements) - numElemInTree
    tree2 = VPTree.createVPTree(elements[0:numElemInTree])
    
    start_time = time.time()
    NNS2 = [VPTree.nearestNeighbour(tree2, elem) for elem in elements[numElemInTree:]]
    total_time = time.time()-start_time
    print("total time:", total_time)
    print("time per element", total_time/elementsChecked)

    printNNS(NNS2, elementsChecked)
    VPTree.toJson(tree2)

# Does not really work. In top ~20%
def lowDimTree(vlmcs, vlmcElements, cutoff):
    elements = []
    elementsToCheck = 6
    for vlmc in vlmcs:
        tree = vlmc.tree['']
        elements.append(VPTreeElement(np.array([tree['A'],tree['C'],tree['G']]), vlmc.name))
    
    numElemInTree = 100 # round(len(elements)*cutoff)
    print(len(elements))

    tree = VPTree.createVPTree(elements[0:numElemInTree])
    NNS = [VPTree.nearestNeighbour(tree, elem) for elem in elements[numElemInTree:numElemInTree+elementsToCheck]]
    printNNS(NNS, elementsToCheck)

    for i in range(elementsToCheck):
        dists = [(node.distance(vlmcElements[numElemInTree+i]), node.identifier) for node in vlmcElements[0:100]]
        dists.sort(key=lambda x:x[0])
        print("current elem: ", vlmcs[numElemInTree+i], "found elem:", NNS[i][0][0][1].value.identifier, "result: \n",dists[0:50], "\n")
        

def printNNS(NNS, elementsChecked):
    i=0
    totalNumberOfActions = 0
    totalDist = 0
    for elem in NNS:
        totalNumberOfActions+=elem[2]
        totalDist+=elem[1]
        i=i+1


    print("avg number of dist calcs", totalNumberOfActions/elementsChecked)
    print("average dist: ", totalDist/elementsChecked)

#from util.parse_vlmc import parse_vlmcs
parser = argparse.ArgumentParser(description="test args")
parser.add_argument("--cutoff", type=float, default=0.5)

add_parse_vlmc_args(parser)
add_distance_arguments(parser)
args = parser.parse_args()

cutoff = args.cutoff
vlmcs = parse_vlmcs(args, "db_config.json")

distance_function = parse_distance_method(args)
elements = [VPTreeVLMC(vlmc, distance_function, vlmc.name) for vlmc in vlmcs]

lowDimTree(vlmcs, elements, cutoff)

#print("number of vlmcs:", len(vlmcs))

#distance_function_stats(elements)
#partOfTree(cutoff, elements)
