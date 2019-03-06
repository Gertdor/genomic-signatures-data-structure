import sys
import time
import argparse
import pickle

import numpy as np
from scipy import stats
sys.path.append('/home/basse/Documents/skola/masterThesis/clustering-genomic-signatures-private')

from clustering_genomic_signatures.util.parse_vlmcs import parse_vlmcs, add_parse_vlmc_args
from clustering_genomic_signatures.util.parse_distance import add_distance_arguments, parse_distance_method
from dataStructures.VPTree import VPTree, VPTreeNode
from dataStructures.VPTreeElement import VPTreeElement
from dataStructures.VLMCElement import VPTreeVLMC
from data_analysis.distance_analysis import distance_function_stats
from data_analysis.distance_function_accuracy import save_distances, load_distances, calculate_pairwise_distance, get_distance_accuracy, calc_pariwise_fast

def fullTree(elements):

    start_time = time.time()
    tree = VPTree.createVPTree(elements)
    NNS = [VPTree.nearestNeighbour(tree, elem) for elem in elements]
    totalTime = time.time() - start_time

    printNNS(NNS)    

def partOfTree(cutoff, elements):
    numElemInTree = round(len(elements)*cutoff)
    elementsChecked = len(elements) - numElemInTree
    tree = VPTree.createVPTree(elements[0:numElemInTree])
    
    start_time = time.time()
    NNS = [VPTree.nearestNeighbour(tree, elem) for elem in elements[numElemInTree:]]
    total_time = time.time()-start_time
    print("total time:", total_time)
    print("time per element", total_time/elementsChecked)

    printNNS(NNS)

def pickleTest(elements):

    tree = VPTree.createVPTree(elements)
    VPTree.save(tree,"test.pickle")
    data = VPTree.load("test.pickle")
    
    print("The trees are equal?:", tree==data)
    


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
    printNNS(NNS)

    for i in range(elementsToCheck):
        dists = [(node.distance(vlmcElements[numElemInTree+i]), node.identifier) for node in vlmcElements[0:100]]
        dists.sort(key=lambda x:x[0])
        print("current elem: ", vlmcs[numElemInTree+i], "found elem:", NNS[i][0][0][1].value.identifier, "result: \n",dists[0:50], "\n")
        

def printNNS(NNS):
    i=0
    totalNumberOfActions = 0
    totalDist = 0
    elementsChecked = len(NNS)
    for elem in NNS:
        totalNumberOfActions+=elem[2]
        totalDist+=elem[1]
        i=i+1

    print("avg number of dist calcs", totalNumberOfActions/elementsChecked)
    print("average dist: ", totalDist/elementsChecked)

def saveDistances(vlmcs, name):
    distances = calc_pariwise_fast(vlmcs)
    save_distances(distances, name)

def calcOverlap(vlmcs):
    tree = VPTree.createVPTree(vlmcs, random=False)
    overlap = VPTree.overlap(tree)
    print(str(overlap))


def testRandom(vlmcs):
    tree1 = VPTree.createVPTree(vlmcs, random=False)
    tree2 = VPTree.createVPTree(vlmcs, random=False)
    tree3 = VPTree.createVPTree(vlmcs, random=True)

    print("These should be true",tree1==tree2)
    print("These should false",tree1==tree3)


#from util.parse_vlmc import parse_vlmcs
parser = argparse.ArgumentParser(description="test args")
parser.add_argument("--cutoff", type=float, default=0.5)

add_parse_vlmc_args(parser)
add_distance_arguments(parser)
args = parser.parse_args()

cutoff = args.cutoff
vlmcs = parse_vlmcs(args, "db_config.json")

vlmcs2 = parse_vlmcs(args, "db_config.json")

distance_function = parse_distance_method(args)
elements = [VPTreeVLMC(vlmc, distance_function, vlmc.name) for vlmc in vlmcs]

#calcOverlap(elements)
#testRandom(elements)
#calcOverlap(elements)

#saveDistances(elements, "data_analysis/small.pickle")

#partOfTree(cutoff, elements)

#pickleTest(elements)

#lowDimTree(vlmcs, elements, cutoff)

#print("number of vlmcs:", len(vlmcs))

#distance_function_stats(elements)
