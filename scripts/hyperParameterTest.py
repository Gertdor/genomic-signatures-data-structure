import argparse
import pickle
from itertools import product
import numpy as np

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
from dataStructures.VLMCElement import VPTreeVLMC
from util.NN_data import NNData
from util.splitElements import split_elements


def hyper_parameter_test(elements, args):
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
    if args.gc_prune_test:
        gc_prune = [True, False]
    else:
        gc_prune = [True]
    if args.k_test:
        k_values = np.arange(args.k_start, args.k_end, args.k_step)
    else:
        k_values = [args.k]

    all_runs = {}
    factors = [p for p in product(greedy_factors, k_values, gc_prune)]
    for factor in factors:
        all_runs[factor] = []
    all_signatures_used = []

    for run_nbr in range(args.number_of_runs):
        print("current run number:", run_nbr)
        (tree_elems, search_elems) = split_elements(elements, args)
        tree = VPTree.createVPTree(
            tree_elems, random=args.random_vp, max_leaf_size=args.leaf_size
        )
        tree_elem_names = [elem.identifier for elem in tree_elems]
        search_elem_names = [elem.identifier for elem in search_elems]
        all_signatures_used.append((tree_elem_names, search_elem_names))
        for factor in factors:
            run_NNS = one_nn_search_run(tree, search_elems, factor)

            run_stats = [nn for nn in run_NNS]
            all_runs[factor].append(run_stats)

    data = NNData(all_runs, all_signatures_used, factors)
    with open(args.o, "wb") as f:
        pickle.dump(data, f)


def one_nn_search_run(tree, search_elems, factors):

    run_NNS = [
        VPTree.nearestNeighbour(tree, elem, factors[1], factors[0], factors[2])
        for elem in search_elems
    ]
    return run_NNS


parser = argparse.ArgumentParser(description="test args")
parser.add_argument(
    "-o", default="greedy_test.pickle", help="output file name of greedy test results"
)
parser.add_argument(
    "--gc_prune_test",
    action="store_true",
    help="should gc distance be used to prune the search results",
)

parser.add_argument(
    "--k_test",
    action="store_true",
    help="should the effect of different k values be tested?",
)
parser.add_argument(
    "--k_start", type=int, default=1, help="initial value of k if running k_test"
)
parser.add_argument(
    "--k_end", default=4, type=int, help="maximum value of k if running k_test"
)
parser.add_argument(
    "--k_step", default=1, type=int, help="step size of k if running k_test"
)

parser.add_argument(
    "--greedy_test",
    action="store_true",
    help=(
        "This will run multiple runs with different greedy factors"
        "determined by --greedy_start, --greedy_end, --greedy_step"
    ),
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
    "--random_vp",
    action="store_true",
    help="Should the vantage point be choosen at random?",
)

parser.add_argument(
    "--cutoff",
    type=float,
    default=0.8,
    help=(
        "How large portion of the data should be stored in the tree."
        "The rest is searched for the nearest neighbour"
    ),
)
parser.add_argument(
    "--leaf_size",
    type=int,
    default=1,
    help="The largest size a leaf may take in the tree",
)
parser.add_argument(
    "--no_randomize_elements",
    action="store_true",
    help="should the elements to be stored/quiered not be randomized",
)

add_parse_vlmc_args(parser)
add_distance_arguments(parser)
args = parser.parse_args()

vlmcs = parse_vlmcs(args, "db_config.json")
print("number of vlmcs:", len(vlmcs))
distance_function = parse_distance_method(args)
tmp = args.distance_function
args.distance_function = "gc-content"
fast_dist = parse_distance_method(args)
args.distance_function = tmp

elements = [
    VPTreeVLMC(vlmc, distance_function, i, fast_dist) for i, vlmc in enumerate(vlmcs)
]

hyper_parameter_test(elements, args)
