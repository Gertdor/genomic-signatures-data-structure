import sys
import time
import argparse
import pickle

import numpy as np
from scipy import stats

sys.path.append(
    "/home/basse/Documents/skola/masterThesis/clustering-genomic-signatures-private"
)

from dataStructures.VPTreeElement import VPTreeElement
from clustering_genomic_signatures.util.parse_vlmcs import (
    parse_vlmcs,
    add_parse_vlmc_args,
)
from clustering_genomic_signatures.util.parse_distance import (
    add_distance_arguments,
    parse_distance_method,
)
from dataStructures.VPTree import VPTree, VPTreeNode
from dataStructures.VPTreeElement import VPTreeElement
from dataStructures.VLMCElement import VPTreeVLMC
from util.distance_util import distance_function_stats
from util.NN_data import NNData

def fullTree(elements, random_element, leaf_size):
    tree = VPTree.createVPTree(elements, random_element, max_leaf_size=leaf_size)
    return tree


def greedy_factor_test(elements, output_file, args):
    """ Perform a test on how the greedy factor effects different metrics

        Input
        -----
        elements - VPTreeVLMC objects that are to be used in the tree, or to be searched for.
        
        output
        ------
        writes the following tuple to the file output_file
        (all_runs, greedy_factors, all_signatures_used)
        they have the following properties
        all_runs -- a list for each greedy factor. That list contains a list of all runs,
                    each run is a list oftuples of length args.num_elem_in_tree or (1-cutoff)*len(elements)
                    whichever is smaller.
                    The tuple has the form: (neighbour ID, distance to neighbor, nbr distance calculations)
        greedy_factors -- a list of all greedy factors used
        all_signatures_used -- a list containing a tuple of form ([elements in tree],[searched points])
    """
    greedy_factors = np.linspace(
        args.greedy_start, args.greedy_end, args.greedy_num_samples
    )
    all_runs = [[] for i in range(args.greedy_num_samples)]
    all_signatures_used = []
    for run_nbr in range(args.number_of_runs):

        print("current run number:", run_nbr)
        (tree_elems, search_elems) = _split_elements(elements, args)
        tree = VPTree.createVPTree(
            tree_elems, random=args.random_vp, max_leaf_size=args.leaf_size
        )
        tree_elem_names = [elem.identifier for elem in tree_elems]
        search_elem_names = [elem.identifier for elem in search_elems]
        all_signatures_used.append((tree_elem_names, search_elem_names))
        for (i, greedy_factor) in enumerate(greedy_factors):

            run_NNS = [
                VPTree.nearestNeighbour(tree, elem, args.k, greedy_factor)
                for elem in search_elems
            ]

            run_stats = [(NN.get_nodes(), NN.get_ops()) for NN in run_NNS]
            all_runs[i].append(run_stats)
    
    data = NNData(all_runs, greedy_factors, all_signatures_used)
    with open(output_file, "wb") as f:
        pickle.dump(data, f)


def multipleNNSearches(elements, args, print_results=True):

    all_NNS = [
        NNSearch(elements, args, print_results=False)
        for i in range(args.number_of_runs)
    ]
    complete_stats = [(NN[1], NN[2]) for NNS in all_NNS for NN in NNS]
    if print_results:
        print(stats.describe(complete_stats))
    return complete_stats


def _split_elements(elems, args):

    if args.randomize_elements:
        elements = elems.copy()
        np.random.shuffle(elements)
    else:
        elements = elems

    num_elem_in_tree = round(len(elements) * args.cutoff)
    if (
        args.number_of_searches == 0
        or args.number_of_searches + num_elem_in_tree > len(elements)
    ):
        args.number_of_searches = len(elements) - num_elem_in_tree

    tree_elements = elements[0:num_elem_in_tree]
    search_elements = elements[
        num_elem_in_tree : num_elem_in_tree + args.number_of_searches
    ]

    return (tree_elements, search_elements)


def NNSearch(elements, args, print_results=True):

    (tree_elements, search_elements) = _split_elements(elements, args)

    elementsChecked = len(search_elements)

    tree = VPTree.createVPTree(
        tree_elements, random=args.random_vp, max_leaf_size=args.leaf_size
    )

    start_time = time.time()
    NNS = [
        VPTree.nearestNeighbour(tree, elem, args.k, args.greedy_factor)
        for elem in search_elements
    ]
    total_time = time.time() - start_time
    if print_results:
        print("total time:", total_time)
        print("time per element", total_time / elementsChecked)
        printNNS(NNS)
    return NNS


# Does not really work. In top ~20%
def lowDimTree(vlmcs, vlmcElements, args):
    elements = []
    elementsToCheck = 6
    for vlmc in vlmcs:
        tree = vlmc.tree[""]
        elements.append(
            VPTreeElement(np.array([tree["A"], tree["C"], tree["G"]]), vlmc.name)
        )

    numElemInTree = 100  # round(len(elements)*args.cutoff)
    print(len(elements))

    tree = VPTree.createVPTree(elements[0:numElemInTree])
    NNS = [
        VPTree.nearestNeighbour(tree, elem, args.k, args.greedy_factor)
        for elem in elements[numElemInTree : numElemInTree + elementsToCheck]
    ]
    printNNS(NNS)

    for i in range(elementsToCheck):
        dists = [
            (node.distance(vlmcElements[numElemInTree + i]), node.identifier)
            for node in vlmcElements[0:100]
        ]
        dists.sort(key=lambda x: x[0])
        print(
            "current elem: ",
            vlmcs[numElemInTree + i],
            "found elem:",
            NNS[i][0][0][1].value.identifier,
            "result: \n",
            dists[0:50],
            "\n",
        )


def printNNS(NNS):
    (avg_dist_calcs, avg_dist) = get_NNS_stats(NNS)
    print("avg number of dist calcs", avg_dist_calcs)
    print("average dist: ", avg_dist)


def get_NNS_stats(NNS):
    i = 0
    totalNumberOfActions = 0
    totalDist = 0
    elementsChecked = len(NNS)
    for elem in NNS:
        totalNumberOfActions += elem.get_ops()
        totalDist += elem.get_cutoff_dist()
        i = i + 1
    return (totalNumberOfActions / elementsChecked, totalDist / elementsChecked)


def generatePointTree(args):
    numberList = [
        VPTreeElement(np.random.uniform(args.min_value, args.max_value, args.dim))
        for x in range(args.number_of_num_searches)
    ]
    tree = VPTree.createVPTree(numberList, args.random_vp, args.leaf_size)
    return tree


def number_NN(tree, args):
    elements = [
        VPTreeElement(np.random.uniform(args.min_value, args.max_value, args.dim))
        for x in range(args.number_of_num_searches)
    ]
    NNS = [
        VPTree.nearestNeighbour(tree, elem, args.k, args.greedy_factor)
        for elem in elements
    ]
    printNNS(NNS)


def calcOverlap(tree):
    overlap = VPTree.overlap(tree)
    print(str(overlap))


parser = argparse.ArgumentParser(description="test args")
parser.add_argument(
    "--cutoff",
    type=float,
    default=0.8,
    help=("How large portion of the data should be stored in the tree."
          "The rest is searched for the nearest neighbour")
)
parser.add_argument(
    "--no_nn_test",
    action="store_true",
    help="Should a nearest neighbour test be performed",
)
parser.add_argument(
    "--overlap",
    action="store_true",
    help="should the overlap in the tree be calculated",
)
parser.add_argument(
    "--low_dim_tree",
    action="store_true",
    help="Should only the ATGC distance be used as distance metric",
)
parser.add_argument(
    "--num",
    action="store_true",
    help="Should whatever tests being run also be run with a tree based on random DIM dimensional points?",
)
parser.add_argument(
    "--no_gs", action="store_true", help="should genomic signatures not be used?"
)
parser.add_argument(
    "--dist_stats",
    action="store_true",
    help=("Should basic statistics be printed on how the distance function"
          "performs for the currently used data set"),
)
parser.add_argument(
    "--leaf_size",
    type=int,
    default=1,
    help="The largest size a leaf may take in the tree",
)
parser.add_argument(
    "--random_vp",
    action="store_true",
    help="Should the vantage point be choosen at random?",
)
parser.add_argument(
    "--dim",
    type=int,
    default=5,
    help="The dimension of the randomly generated numbers"
)
parser.add_argument(
    "--max_value",
    type=int,
    default=5,
    help="max value of the randomly generated numbers",
)
parser.add_argument(
    "--min_value",
    type=int,
    default=0,
    help="min value of the randomly generated numbers",
)
parser.add_argument(
    "--number_of_points",
    type=int,
    default=1000,
    help="Number of DIM dimensional points to generate",
)
parser.add_argument(
    "--number_of_num_searches",
    type=int,
    default=100,
    help="Number of DIM dimensional points to search for nearest neighbours with",
)
parser.add_argument(
    "--greedy_factor",
    type=float,
    default=1,
    help=("Determines how greedy the pruning should be. A solution must be atleast this much"
          "better to be considered. Default is 1, that is, anything which has the ability to be better"
          "is considered. 2 means the solution has to be atleast twice as good to be considered"),
)
parser.add_argument(
    "--k",
    type=int,
    default=1,
    help="how many neighbours should be found? default=1"
)
parser.add_argument(
    "--randomize_elements",
    action="store_true",
    help="should the elements to be stored/quiered be randomized",
)
parser.add_argument(
    "--number_of_runs",
    type=int,
    default=1,
    help="How many times should an NN search be repeated for a specific setting.",
)
parser.add_argument(
    "--number_of_searches",
    type=int,
    default=0,
    help="how many of the elements should be used to NN search in the tree?",
)

parser.add_argument(
    "--greedy_test",
    action="store_true",
    help=("This will run multiple runs with different greedy factors"
          "determined by --greedy_start, --greedy_end, --greedy_step"),
)
parser.add_argument(
    "--greedy_start",
    type=float,
    default=1,
    help="start value of greedy_factor when running greedy__test",
)
parser.add_argument(
    "--greedy_end",
    type=float,
    default=5,
    help="end value of greedy_factor when running greedy_test",
)
parser.add_argument(
    "--greedy_num_samples",
    type=int,
    default=21,
    help="number of samples to take between greedy_start and greedy_end when running greedy_test",
)
parser.add_argument(
    "--greedy_test_file_name",
    default="greedy_test_data.pickle",
    help="where should the data be saved after the greedy_factor_test has been run",
)
parser.add_argument(
    "-o", default="greedy_test.pickle", help="output file name of greedy test results"
)

add_parse_vlmc_args(parser)
add_distance_arguments(parser)
args = parser.parse_args()

vlmcs = parse_vlmcs(args, "db_config.json")
print("number of vlmcs:",len(vlmcs))
distance_function = parse_distance_method(args)
elements = [VPTreeVLMC(vlmc, distance_function, i) for i, vlmc in enumerate(vlmcs)]

if args.overlap:
    if args.num:
        point_tree = generatePointTree(args)
        print("point tree overlap")
        calcOverlap(point_tree)
    if not args.no_gs:
        vlmc_tree = fullTree(elements, args.random_vp, args.leaf_size)
        print("vlmc tree overlap")
        calcOverlap(vlmc_tree)


if not args.no_nn_test:
    if args.num:
        point_tree = generatePointTree(args)
        number_NN(point_tree, args)
    if not args.no_gs:
        if args.number_of_runs == 1:
            NNSearch(elements, args)
        else:
            multipleNNSearches(elements, args)

if args.greedy_test:
    greedy_factor_test(elements, "greedy_factor_test.pickle", args)

if args.low_dim_tree:
    lowDimTree(vlmcs, elements)

if args.dist_stats:
    distance_function_stats(elements)
