import sys
import time
import argparse
import pickle
import pytest

import numpy as np
from scipy import stats

sys.path.append(
    "/home/basse/Documents/skola/masterThesis/clustering-genomic-signatures-private"
)

from clustering_genomic_signatures.util.parse_signatures import (
    parse_signatures,
    add_parse_signature_args,
)
from clustering_genomic_signatures.util.parse_distance import (
    add_distance_arguments,
    parse_distance_method,
)
from dataStructures.VPTree import VPTree, VPTreeNode
from dataStructures.VPTreeElement import VPTreeElement
from dataStructures.VLMCElement import VPTreeVLMC

# from data_analysis.distance_analysis import distance_function_stats
# from data_analysis.distance_function_accuracy import (
#    save_distances,
#    load_distances,
#    calculate_pairwise_distance,
#    get_distance_accuracy,
#    calc_pariwise_fast,
# )


def test_pickle(elements):

    tree = VPTree(elements)
    VPTree.save(tree, "test.pickle")
    data = VPTree.load("test.pickle")
    assert tree == data


def test_overlap(vlmcs):
    tree = VPTree.createVPTree(vlmcs, random=False)
    overlap = VPTree.overlap(tree)
    assert overlap == 3738


def test_return_type(vlmcs):
    tree = VPTree.createVpTree(elements)
    NNS = [VPTree.nearestNeighbour(tree, elem) for elem in vlmcs]
    for nn in NNS:
        assert type(nn[0][0][1]) is VPTreeVLMC


def test_random(vlmcs):
    tree1 = VPTree.createVPTree(vlmcs, random=False)
    tree2 = VPTree.createVPTree(vlmcs, random=False)
    tree3 = VPTree.createVPTree(vlmcs, random=True)

    assert tree1 == tree2
    assert tree1 != tree3


def test_leaf_size():
    numberList = [VPTreeElement(np.random.uniform(0, 4.5, 5)) for x in range(1000)]
    tree1 = VPTree.createVPTree(numberList, False, 1)
    tree2 = VPTree.createVPTree(numberList, False, 6)
    NN1 = VPTree.nearestNeighbour(tree1, VPTreeElement(np.array([2, 4, 3, 2, 2])))
    NN2 = VPTree.nearestNeighbour(tree2, VPTreeElement(np.array([2, 4, 3, 2, 2])))
    assert (
        NN1[0] == NN2[0]
    )  # Dont want to compare the number of lookups, just the element found


# from util.parse_vlmc import parse_vlmcs
parser = argparse.ArgumentParser(description="test args")
parser.add_argument("--cutoff", type=float, default=0.5)

add_parse_signature_args(parser)
add_distance_arguments(parser)
args = parser.parse_args()

cutoff = args.cutoff
vlmcs = parse_signatures(args, "../gs-data-structures/db_config.json")

vlmcs2 = parse_signatures(args, "../gs-data-structures/db_config.json")

distance_function = parse_distance_method(args)
elements = [VPTreeVLMC(vlmc, distance_function, vlmc.name) for vlmc in vlmcs]

test_pickle(elements)
# test_overlap(elements)
# test_random(elements)
# test_leaf_size()
# test_return_type(elements)
