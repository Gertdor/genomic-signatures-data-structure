import sys
import time

sys.path.append('/home/basse/Documents/skola/masterThesis/clustering-genomic-signatures-private')

from clustering_genomic_signatures.util import parse_vlmcs, add_parse_vlmc_args

import argparse
from dataStructures.VPTree import VPTree
from dataStructures.VLMCElement import VPTreeVLMC


def fullTree(elements):
    tree = VPTree.createVPTree(elements)
    NNS = [VPTree.nearestNeighbour(tree, elem) for elem in elements]
    totalTime = start_time - time.time()
    
    print(NNS[0])
    i = 0
    count = 0
    totalNumberOfActions = 0
    for elem in NNS:
        totalNumberOfActions+=elem[2]
        if(elem[0][0][1].value == elements[i]):
            count=count+1
        i=i+1
    
    print(count)
    print(totalNumberOfActions/len(vlmcs))

def partOfTree(cutoff, elements):
    numElem = len(elements)-cutoff
    tree2 = VPTree.createVPTree(elements[0:cutoff])
    
    start_time = time.time()
    NNS2 = [VPTree.nearestNeighbour(tree2, elem) for elem in elements[cutoff:]]
    total_time = time.time()-start_time
    
    print("total time:", total_time)
    print("time per element", total_time/numElem)
    i=0
    totalNumberOfActions = 0
    totalDist = 0
    for elem in NNS2:
        totalNumberOfActions+=elem[2]
        totalDist+=elem[1]
        i=i+1
    
    print("avg number of dist calcs", totalNumberOfActions/numElem)
    print("average dist: ", totalDist/numElem)


#from util.parse_vlmc import parse_vlmcs
parser = argparse.ArgumentParser(description="test args")

add_parse_vlmc_args(parser)

args = parser.parse_args()

vlmcs = parse_vlmcs(args, "db_config.json")

elements = [VPTreeVLMC(vlmc) for vlmc in vlmcs]

partOfTree(5000, elements)
