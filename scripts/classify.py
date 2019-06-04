import argparse
import pickle
import time

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
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)
from julia import Main, PstClassifier
import sys


def get_files(args):
    if isfile(args.input):
        files = [args.input]
    else:
        files = [join(args.input,f) for f in listdir(args.input)]

    return(files)


def NN_search_on_file(fna_file, dist_fun, fast_dist):
    
    sequences = SeqIO.parse(fna_file, "fasta")
    sequences = [(fna_file, str(value.seq)) for value in sequences]

    VLMCParseTree = VLMC.train_multiple(
        sequences, args.max_depth, args.min_count, args.VLMC_free_parameters
    )
     
    VLMCs = [
        VLMC.from_tree(tree.split("\n"), name=name) for name, tree in VLMCParseTree
    ]

    VPTreeElements = [
        VPTreeVLMC(signature, dist_fun, i, fast_dist)
        for i, signature in enumerate(VLMCs)
    ]
    
    NNS = [
        tree.nearest_neighbor(element, args.k, args.greedy_factor, args.no_gc_prune)
        for element in VPTreeElements
    ]

    return NNS


parser = argparse.ArgumentParser(description="parser for classification algorithm")

parser.add_argument(
    "--tree", help="Path to the pickled vantage point tree", required=True
)
parser.add_argument(
    "--input", help="path to input file or folder", required=True
)
parser.add_argument(
    "--min_count", type=int, default=4, help="minimum frequency of kmer"
)
parser.add_argument(
    "--VLMC_free_parameters", default=100, type=int, help="number of parameters of VLMC"
)
parser.add_argument(
    "--max_depth", default=5, type=int, help="maximum branch depth of VLMC"
)
parser.add_argument("-k", default=1, type=int, help="number of neighbors to find")
parser.add_argument(
    "--no_gc_prune", action="store_false", help="Should gc pruning be disabled"
)
parser.add_argument(
    "--greedy_factor", default=1, type=float, help="greedy factor to be used"
)
parser.add_argument(
    "--rank",
    default="genus",
    choices=["order", "family", "genus", "species"],
    help="at which taxonomic rank should the classification be made?",
)
parser.add_argument(
    "--one_class",
    action="store_true",
    help=(
        "should the most probable class be returned. Default is to return a dictionary "
        + " with weights for all classes found"
    ),
)
parser.add_argument(
    "-o",
    default="classification_result.pickle",
    help="name of file containing the classifications",
)

add_parse_signature_args(parser)
add_distance_arguments(parser)

args = parser.parse_args()
tree = VPTree.load(args.tree)

fna_files = get_files(args)

dist_fun = parse_distance_method(args)
args.distance_function = "gc-content"
fast_dist = parse_distance_method(args)

start_time = time.time()

sequences_classified = 0
classifications = {}
for i,fna_file in enumerate(fna_files):
    if(not i%10):
        print("time elapsed: " + str(start_time - time.time()))
        print("sequences classified so far: " + str(sequences_classified))
        print(i)

    NNS = NN_search_on_file(fna_file, dist_fun, fast_dist)
    sequences_classified+=len(NNS)

    classifications[fna_file] = [nn.classify(args.rank) for nn in NNS]

with open(args.o, "wb") as f:
    pickle.dump(classifications, f)

print("total number of sequences: " + str(sequences_classified))
print("total time taken " + str(start_time-time.time()))
print("wrote data to: " + args.o)
