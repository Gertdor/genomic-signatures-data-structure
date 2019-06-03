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

def create_signatures(args):
    if isfile(args.input):
        files = [args.input]
    else:
        files = [join(args.input,f) for f in listdir(args.input)]

    sequences = [SeqIO.parse(fna_file, "fasta") for fna_file in files]
    sequences = [
        [(name, str(value.seq)) for value in s] for name, s in zip(files, sequences)
    ]

    total_len = sum(len(l) for l in sequences)
    print(total_len)
    
    VLMCParseTrees = []
    for i,sequence in enumerate(sequences):
        if(not i%10):
            print("files completed: " + str(i))
        VLMCParseTrees.append(VLMC.train_multiple(
            sequence, args.max_depth, args.min_count, args.VLMC_free_parameters
        ))

    VLMCs = [[
        VLMC.from_tree(tree.split("\n"), name=name) for name, tree in VLMCParseTree
    ] for VLMCParseTree in VLMCParseTrees]
    return (files,VLMCs)


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

print(args.o)

start = time.time()
names, signatures_list = create_signatures(args)

end_seq_time = time.time()
print("it took " + str(end_seq_time-start) + " to generate sequences")

dist_fun = parse_distance_method(args)
args.distance_function = "gc-content"
fast_dist = parse_distance_method(args)

VPTreeElements_list = [[
    VPTreeVLMC(signature, dist_fun, i, fast_dist)
    for i, signature in enumerate(signatures)
] for signatures in signatures_list]

done_VLMC_generate = time.time()
print("it took " + str(done_VLMC_generate - end_seq_time) + " to create tree elements")

all_NNS = []

for i,VPTreeElements in enumerate(VPTreeElements_list):
    if(not i%10):
        print("number of files classified: " + str(i))
    all_NNS.append([
        tree.nearest_neighbor(element, args.k, args.greedy_factor, args.no_gc_prune)
        for element in VPTreeElements
    ])

classification = {name:[(NN.classify(args.rank)) for NN in NNS] for name, NNS in zip(names,all_NNS)}

done_classify = time.time()

print("it took " + str(done_classify - done_VLMC_generate) + " to classify vlmc")

with open(args.o, "wb") as f:
    pickle.dump(classification, f)

print("wrote data to: " + args.o)
