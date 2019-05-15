import argparse
import pickle

from dataStructures.VPForest import VPForest
from dataStructures.VPTreeElement import VPTreeElement
from dataStructures.VPTree import VPTree, VPTreeNode
from util.generateVLMCElement import generate_vlmc_elements, add_generate_vlmc_args

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

add_generate_vlmc_args(parser)

args = parser.parse_args()
elements = generate_vlmc_elements(args)

if(args.forest):
    tree = VPForest(elements, random=args.random_vp, max_leaf_size = args.leaf_size)
else:
    tree = VPTree(elements, random=args.random_vp, max_leaf_size = args.leaf_size)

print("saving tree in: " + args.o)
tree.save(args.o)
