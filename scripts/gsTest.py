import sys
import time
import argparse
import pickle

import numpy as np
from scipy import stats
sys.path.append('/home/basse/Documents/skola/masterThesis/clustering-genomic-signatures-private')

from dataStructures.VPTreeElement import VPTreeElement
from clustering_genomic_signatures.util.parse_vlmcs import parse_vlmcs, add_parse_vlmc_args
from clustering_genomic_signatures.util.parse_distance import add_distance_arguments, parse_distance_method
from dataStructures.VPTree import VPTree, VPTreeNode
from dataStructures.VPTreeElement import VPTreeElement
from dataStructures.VLMCElement import VPTreeVLMC
from data_analysis.distance_analysis import distance_function_stats
from data_analysis.distance_function_accuracy import save_distances, load_distances, calculate_pairwise_distance, get_distance_accuracy, calc_pariwise_fast

def fullTree(elements, random_element, leaf_size):
    tree = VPTree.createVPTree(elements, random_element, max_leaf_size=leaf_size)
    return(tree)

def partOfTree(cutoff, elements, random, leaf_size):
    numElemInTree = round(len(elements)*cutoff)
    elementsChecked = len(elements) - numElemInTree
    tree = VPTree.createVPTree(elements[0:numElemInTree],random=random, max_leaf_size = leaf_size)
    
    start_time = time.time()
    NNS = [VPTree.nearestNeighbour(tree, elem) for elem in elements[numElemInTree:]]
    total_time = time.time()-start_time
    print("total time:", total_time)
    print("time per element", total_time/elementsChecked)

    printNNS(NNS)


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


def generatePointTree(args):
    numberList = [VPTreeElement(np.random.uniform(args.min_value, args.max_value, args.dim )) for x in range(args.number_of_searches)]
    tree = VPTree.createVPTree(numberList, args.random_vp, args.leaf_size)
    return(tree)

def number_NN(tree, args):
    elements = [VPTreeElement(np.random.uniform(args.min_value,args.max_value, args.dim)) for x in range(args.number_of_searches)]
    NNS = [VPTree.nearestNeighbour(tree,elem,1) for elem in elements]
    printNNS(NNS)

def saveDistances(vlmcs, name):
    distances = calc_pariwise_fast(vlmcs)
    save_distances(distances, name)


def calcOverlap(tree):
    overlap = VPTree.overlap(tree)
    print(str(overlap))


parser = argparse.ArgumentParser(description="test args")
parser.add_argument("--cutoff", type=float, default=0.5)
parser.add_argument("--nn_test", action='store_true')
parser.add_argument("--overlap", action='store_true')
parser.add_argument("--low_dim_tree", action='store_true')
parser.add_argument("--num",action='store_true')

parser.add_argument("--dist_stats", action='store_true')
parser.add_argument("--leaf_size", type=int, default=1)
parser.add_argument("--random_vp",action='store_true')
parser.add_argument("--print",action='store_true')
parser.add_argument("--dim", type=int, default=5)
parser.add_argument("--max_value", type=int, default=5)
parser.add_argument("--min_value", type=int, default=0)
parser.add_argument("--number_of_points", type=int, default=1000)
parser.add_argument("--number_of_searches", type=int, default=100)

add_parse_vlmc_args(parser)
add_distance_arguments(parser)
args = parser.parse_args()

vlmcs = parse_vlmcs(args, "db_config.json")
distance_function = parse_distance_method(args)
elements = [VPTreeVLMC(vlmc, distance_function, vlmc.name) for vlmc in vlmcs]

if(args.overlap):
    if(args.num):
        point_tree = generatePointTree(args)
        print("point tree overlap")
        calcOverlap(point_tree)
    else:
        vlmc_tree = fullTree(elements,args.random_vp,args.leaf_size)
        print("vlmc tree overlap")
        calcOverlap(vlmc_tree)

if(args.nn_test):
    if(args.num):
        point_tree = generatePointTree(args)
        number_NN(point_tree,args)
    else:
        cutoff = args.cutoff
        partOfTree(cutoff, elements, args.random_vp, args.leaf_size)

if(args.low_dim_tree):
    cutoff = args.cutoff
    lowDimTree(vlmcs, elements, cutoff)

if(args.dist_stats):
    distance_function_stats(elements)
