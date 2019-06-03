import argparse
import pickle
from itertools import product
import numpy as np
import time

from dataStructures.VPForest import VPForest
from dataStructures.VPTreeElement import VPTreeElement
from dataStructures.VPTree import VPTree, VPTreeNode
from dataStructures.VLMCElement import VPTreeVLMC

from util.NN_data import NNData
from util.generateVLMCElements import generate_vlmc_elements, add_generate_vlmc_args

from sklearn.model_selection import RepeatedKFold

def hyper_parameter_test(elements, args):
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
    splitter = RepeatedKFold(args.n_split, args.n_repeat, random_state = args.random_seed)
    i = 0
    for tree_indexes, search_indexes in splitter.split(elements):
        print("current run number:", i)
        i+=1
        tree_elems = elements[tree_indexes]
        search_elems = elements[search_indexes]
        
        if args.forest:
            forest = VPForest(
                tree_elems, random=args.random_vp, max_leaf_size=args.leaf_size
            )
        tree = VPTree(tree_elems, random=args.random_vp, max_leaf_size=args.leaf_size)
        tree_elem_names = [elem.identifier for elem in tree_elems]
        search_elem_names = [elem.identifier for elem in search_elems]
        all_signatures_used.append((tree_elem_names, search_elem_names))
        start = time.time()
        for factor in factors:
            if factor[0]:
                run_NNS = one_nn_search_run(forest, search_elems, factor, args.parallel)
            else:
                run_NNS = one_nn_search_run(tree, search_elems, factor, args.parallel)
            all_runs[factor].append(run_NNS)

        print("search time:", time.time()-start)
    data = NNData(all_runs, all_signatures_used, factors)
    with open(args.o, "wb") as f:
        pickle.dump(data, f)


def one_nn_search_run(tree, search_elems, factors, parallel):
    if parallel:
        run_NNS = tree.many_nearest_neighbor(
            search_elems, factors[2], factors[1], factors[3], args.pool_size
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
    "--pool_size",
    type=int,
    default=3,
    help="size of worker pool for parallel execution"
)

parser.add_argument(
    "--greedy_start",
    type=float,
    default=1,
    help="start value of greedy_factor when running greedy__test",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=None,
    help="the random seed used to split the data",
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
    "--n_repeat",
    type=int,
    default=1,
    help="How many times should the entire dataset be used. Used for sklearn's RepeatedKFold",
)

parser.add_argument(
    "--n_split",
    type=int,
    default=5,
    help="how many portions should the dataset be split into. Used for sklearn's RepeatedKFold"
)

parser.add_argument(
    "--random_vp",
    action="store_true",
    help="Should the vantage point be choosen at random?",
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
add_generate_vlmc_args(parser)
args = parser.parse_args()
elements = np.array(generate_vlmc_elements(args))
print("number of VLMC: " + str(len(elements)))
start_time = time.time()
hyper_parameter_test(elements, args)
print("time:", time.time() - start_time)
