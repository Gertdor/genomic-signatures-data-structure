import argparse

from os import listdir
from os.path import isfile, join

from Bio import SeqIO, Entrez

from clustering_genomic_signatures.util.parse_signatures import (
    parse_signatures,
    add_parse_signature_args,
)
from clustering_genomic_signatures.util.parse_distance import (
    add_distance_arguments,
    parse_distance_method,
)
from clustering_genomic_signatures.signature import VLMC

from dataStructures.VPTree import VPTree, VPTreeNode
from dataStructures.VLMCElement import VPTreeVLMC
from util.NN_data import NNData
from util.splitElements import split_elements
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)


def create_signatures(args):
    if isfile(args.input):
        files = [args.input]
    else:
        files = [f for f in listdir(args.input) if isfile(f)]

    sequences = [SeqIO.parse(fna_file, "fasta") for fna_file in files]
    sequences = [
        (str(i), str(value.seq)) for i, s in enumerate(sequences) for value in s
    ]
    VLMCParseTree = VLMC.train_multiple(
        sequences, args.max_depth, args.min_count, args.VLMC_free_parameters
    )

    VLMCs = [
        VLMC.from_tree(tree, name=name) for name, tree in zip(args.input, VLMCParseTree)
    ]
    return VLMCs


parser = argparse.ArgumentParser(description="parser for classifcation algorithm")

parser.add_argument("--tree", help="Path to the pickled vantage point tree", required=True)
parser.add_argument("--input", help="path to input file or folder", required=True)
parser.add_argument(
    "--min_count", type=int, default=10, help="minimum frequency of kmer"
)
parser.add_argument(
    "--VLMC_free_parameters", default=100, type=int, help="number of parameters of VLMC"
)
parser.add_argument(
    "--max_depth", default=15, type=int, help="maximum branch depth of VLMC"
)
parser.add_argument("--neighbors", type=int, help="number of neighbors to find")
parser.add_argument(
    "--no_gc_prune", action="store_false", help="Should gc pruning be disabled"
)
parser.add_argument("--greedy_factor", type=float, help="greedy factor to be used")

add_parse_signature_args(parser)
add_distance_arguments(parser)

args = parser.parse_args()

tree = VPTree.load(args.tree)

signatures = create_signatures(args)
print("number of generated signatures" + str(len(signatures)))
dist_fun = parse_distance_method(args)
args.distance_function = "gc-content"
fast_dist = parse_distance_method(args)

VPTreeElements = [
    VPTreeVLMC(signature, dist_fun, i, fast_dist)
    for i, signature in enumerate(signatures)
]

NNS = [
    tree.nearest_neighbor(element, args.neighbors, args.greedy_factor, args.gc_prune)
    for element in VPTreeElements
]
