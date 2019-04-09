import argparse
import pickle
from itertools import product
import numpy as np
from multiprocessing import Pool
import time

from dataStructures.VPForest import VPForest
from dataStructures.VPTreeElement import VPTreeElement
from clustering_genomic_signatures.util.parse_signatures import (
    parse_signatures,
    add_parse_signature_args,
)
from clustering_genomic_signatures.util.parse_distance import (
    add_distance_arguments,
    parse_distance_method,
)
from dataStructures.VPTree import VPTree, VPTreeNode
from dataStructures.VLMCElement import VPTreeVLMC
from util.NN_data import NNData
from util.splitElements import split_elements
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)


def hyper_parameter_test(elements, meta_data, args):
    """ Perform a test on how the greedy factor effects different metrics

        Input
        -----
        elements - VPTreeVLMC objects that are to be used in the tree, or to be searched for.
        
        output
        ------
        writes an NNData object to the file location of args.o
    """

    greedy_factors = np.linspace(
        args.greedy_start, args.greedy_end, args.greedy_num_samples
    )
    print("greedy:", greedy_factors)
    k_values = np.arange(args.k_start - 1, args.k_end, args.k_step) + 1
    k_values = [int(k) for k in k_values]
    if args.gc_prune_test:
        gc_prune = [True, False]
    else:
        gc_prune = [True]
    if args.forest:
        forest = [True, False]
    else:
        forest = [False]

    all_runs = {}
    factors = [p for p in product(forest, greedy_factors, k_values, gc_prune)]
    for factor in factors:
        all_runs[factor] = []
    all_signatures_used = []

    for run_nbr in range(args.number_of_runs):
        print("current run number:", run_nbr)
        (tree_elems, search_elems) = split_elements(elements, args)
        if args.forest:
            forest = VPForest(
                tree_elems, random=args.random_vp, max_leaf_size=args.leaf_size
            )
        tree = VPTree(tree_elems, random=args.random_vp, max_leaf_size=args.leaf_size)
        tree_elem_names = [elem.identifier for elem in tree_elems]
        search_elem_names = [elem.identifier for elem in search_elems]
        all_signatures_used.append((tree_elem_names, search_elem_names))
        for factor in factors:
            if factor[0]:
                run_NNS = one_nn_search_run(forest, search_elems, factor, args.parallel)
            else:
                run_NNS = one_nn_search_run(tree, search_elems, factor, args.parallel)
            all_runs[factor].append(run_NNS)

    data = NNData(all_runs, all_signatures_used, factors, meta_data)
    with open(args.o, "wb") as f:
        pickle.dump(data, f)


def one_nn_search_run(tree, search_elems, factors, parallel):
    if parallel:
        print("whathaht")
        run_NNS = tree.many_nearest_neighbor(
            search_elems, factors[2], factors[1], factors[3]
        )
    else:
        run_NNS = [
            tree.nearest_neighbor(elem, factors[2], factors[1], factors[3])
            for elem in search_elems
        ]
    return run_NNS


parser = argparse.ArgumentParser(description="test args")

parser.add_argument(
    "-o",
    default="hyper_parameter_test.pickle",
    help="output file name of greedy test results",
)
parser.add_argument(
    "--gc_prune_test",
    action="store_true",
    help="should gc distance be used to prune the search results",
)

parser.add_argument("--k_start", type=int, default=1, help="initial value of k")
parser.add_argument("--k_end", default=1, type=int, help="maximum value of k")
parser.add_argument("--k_step", default=1, type=int, help="step size of k")

parser.add_argument(
    "--greedy_start",
    type=float,
    default=1,
    help="start value of greedy_factor when running greedy__test",
)
parser.add_argument(
    "--greedy_end",
    type=float,
    default=1,
    help="end value of greedy_factor when running greedy_test",
)
parser.add_argument(
    "--greedy_num_samples",
    type=int,
    default=1,
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
parser.add_argument(
    "--parallel", action="store_true", help="Should the nn be searched for in parallel?"
)

parser.add_argument(
    "--forest", action="store_true", help="Should a VPForest be used instead of VPTree"
)

add_parse_signature_args(parser)
add_distance_arguments(parser)
args = parser.parse_args()
db_config_path = "db_config.json"

vlmcs = parse_signatures(args, db_config_path)
names = [vlmc.name for vlmc in vlmcs]
meta_data = get_metadata_for(names, db_config_path)
print("number of vlmcs:", len(vlmcs))
distance_function = parse_distance_method(args)
tmp = args.distance_function
args.distance_function = "gc-content"
fast_dist = parse_distance_method(args)
args.distance_function = tmp

elements = [
    VPTreeVLMC(vlmc, distance_function, i, fast_dist) for i, vlmc in enumerate(vlmcs)
]

start_time = time.time()
hyper_parameter_test(elements, meta_data, args)
print("time:", time.time() - start_time)
