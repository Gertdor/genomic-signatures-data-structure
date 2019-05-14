import argparse
import pickle

from dataStructures.VPForest import VPForest
from dataStructures.VPTreeElement import VPTreeElement
from dataStructures.VPTree import VPTree, VPTreeNode
from dataStructures.VLMCElement import VPTreeVLMC
from util.subsetMetadata import subset_all_metadata

from clustering_genomic_signatures.util.parse_signatures import (
    parse_signatures,
    add_parse_signature_args,
)
from clustering_genomic_signatures.util.parse_distance import (
    add_distance_arguments,
    parse_distance_method,
)
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)


parser = argparse.ArgumentParser("parser for create vp tree")

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
    "--forest", action="store_true", help="Should a VPForest be used instead of VPTree"
)
parser.add_argument(
    "-o",default="tree.pickle",help="filename where the pickled tree is stored"
)

add_parse_signature_args(parser)
add_distance_arguments(parser)

args = parser.parse_args()
db_config_path = "db_config.json"

vlmcs = parse_signatures(args, db_config_path)
names = [vlmc.name for vlmc in vlmcs]
meta_data = subset_all_metadata(get_metadata_for(names, db_config_path))

distance_function = parse_distance_method(args)
tmp = args.distance_function
args.distance_function = "gc-content"
fast_dist = parse_distance_method(args)
args.distance_function = tmp

elements = [
    VPTreeVLMC(vlmc, distance_function, fast_dist = fast_dist, taxonomic_data = meta_data[name])
    for name,vlmc in zip(names,vlmcs)
]

if(args.forest):
    tree = VPForest(elements, random=args.random_vp, max_leaf_size = args.leaf_size)
else:
    tree = VPTree(elements, random=args.random_vp, max_leaf_size = args.leaf_size)

print("saving tree in: " + args.o)
tree.save(args.o)
