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

def partOfTree(elems, args):
    if(args.randomize_elements):
        elements = elems.copy()
        np.random.shuffle(elements)
    else:
        elements = elems
    numElemInTree = round(len(elements)*args.cutoff)
    elementsChecked = len(elements) - numElemInTree
    tree = VPTree.createVPTree(elements[0:numElemInTree],random=args.random_vp, max_leaf_size = args.leaf_size)
    
    start_time = time.time()
    NNS = [VPTree.nearestNeighbour(tree, elem, args.k, args.greedy_factor) for elem in elements[numElemInTree:]]
    total_time = time.time()-start_time
    print("total time:", total_time)
    print("time per element", total_time/elementsChecked)

    printNNS(NNS)
    resultQuality(elements[0:numElemInTree],NNS)

# Does not really work. In top ~20%
def lowDimTree(vlmcs, vlmcElements, args):
    elements = []
    elementsToCheck = 6
    for vlmc in vlmcs:
        tree = vlmc.tree['']
        elements.append(VPTreeElement(np.array([tree['A'],tree['C'],tree['G']]), vlmc.name))
    
    numElemInTree = 100 # round(len(elements)*args.cutoff)
    print(len(elements))

    tree = VPTree.createVPTree(elements[0:numElemInTree])
    NNS = [VPTree.nearestNeighbour(tree, elem, args.k, args.greedy_factor) for elem in elements[numElemInTree:numElemInTree+elementsToCheck]]
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
    NNS = [VPTree.nearestNeighbour(tree,elem,args.k,args.greedy_factor) for elem in elements]
    printNNS(NNS)

def saveDistances(vlmcs, name):
    distances = calc_pariwise_fast(vlmcs)
    save_distances(distances, name)

def resultQuality(points_in_tree, NNS):
    #distances = [[tree_node.distance(NN[0][0][1]) for tree_node in points_in_tree] for NN in NNS]
    #print(min(distances))
    return 0

def calcOverlap(tree):
    overlap = VPTree.overlap(tree)
    print(str(overlap))


parser = argparse.ArgumentParser(description="test args")
parser.add_argument("--cutoff", type=float, default=0.8, help="How large portion of the data should be stored in the tree. The rest is searched for the nearest neighbour")
parser.add_argument("--nn_test", action='store_false',help= "Should a nearest neighbour test be performed")
parser.add_argument("--overlap", action='store_true',help= "should the overlap in the tree be calculated")
parser.add_argument("--low_dim_tree", action='store_true', help="Should only the ATGC distance be used as distance metric")
parser.add_argument("--num",action='store_true',help="Should whatever tests being run also be run with a tree based on random DIM dimensional points?")
parser.add_argument("--no_gs",action='store_true',help="should genomic signatures not be used?")
parser.add_argument("--dist_stats", action='store_true', help="Should basic statistics be printed on how the distance function performs for the currently used data set")
parser.add_argument("--leaf_size", type=int, default=1, help="The largest size a leaf may take in the tree")
parser.add_argument("--random_vp",action='store_true', help="Should the vantage point be choosen at random?")
parser.add_argument("--dim", type=int, default=5, help="The dimension of the randomly generated numbers")
parser.add_argument("--max_value", type=int, default=5, help="max value of the randomly generated numbers")
parser.add_argument("--min_value", type=int, default=0, help="min value of the randomly generated numbers")
parser.add_argument("--number_of_points", type=int, default=1000, help="Number of DIM dimensional points to generate")
parser.add_argument("--number_of_searches", type=int, default=100, help="Number of DIM dimensional points to search for nearest neighbours with")
parser.add_argument("--greedy_factor",type=float, default=0, help="Determines how greedy the pruning is. A solution must be atleast this much better to be considered. Default is 0, that is, anything which has the ability to be better is considered.")
parser.add_argument("--k",type=int,default=1, help="how many neighbours should be found? default=1")
parser.add_argument("--randomize_elements",action='store_true',help="should the elements to be stored/quiered be randomized")

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
    if(not args.no_gs):
        vlmc_tree = fullTree(elements,args.random_vp,args.leaf_size)
        print("vlmc tree overlap")
        calcOverlap(vlmc_tree)

if(args.nn_test):
    if(args.num):
        point_tree = generatePointTree(args)
        number_NN(point_tree,args)
    if(not args.no_gs):
        partOfTree(elements, args)

if(args.low_dim_tree):
    lowDimTree(vlmcs, elements)

if(args.dist_stats):
    distance_function_stats(elements)
