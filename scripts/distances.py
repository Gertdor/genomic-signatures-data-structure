import argparse
import pickle
import numpy as np

from operator import itemgetter
from scipy import stats
from dataStructures.VLMCElement import VPTreeVLMC
from operator import itemgetter

from clustering_genomic_signatures.util.parse_vlmcs import (
    parse_vlmcs,
    add_parse_vlmc_args,
)
from clustering_genomic_signatures.util.parse_distance import (
    add_distance_arguments,
    parse_distance_method,
)


def save_neighbor_order(distances, filename):
    with open(filename, "wb") as f:
        pickle.dump(distances, f)


def load_neighbor_order(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def calculate_neighbor_order(vlmcs, distance_function):
    names = np.array([vlmc.name for vlmc in vlmcs])
    distances = distance_function.distances(vlmcs)
    neighbor_order = np.argsort(distances, 1)
    return (np.array(neighbor_order), names)

def norm_to_gc_dist(vlmcs, neighbor_order, number_of_neighbors, filename=None):
    """ calculate how the frobenius-norm distance compares to GC content

        input
        -----
        vlmcs -- a list of VLMCs
        neighbor_order -- matrix containing the order of neighbors
                          calculated with the calculate_neighbor_order function
        number_of_neighbors -- the number of neighbors to examine for each vlmc
        filename -- optional, if filename is None then the results are returned,
                    otherwise they are pickled to the file 'filename'

        returns
        ------
        A list of tuples on the form: [(norm-distance,GC distance)]
    """
    args.distance_function = 'frobenius-norm'
    frobenius_norm = parse_distance_method(args)
    args.distance_function = 'gc-content'
    gc_function = parse_distance_method(args)
    
    all_distances = []
    for i,vlmc in enumerate(vlmcs):
        for k in range(number_of_neighbors):
            frobenius_distance = frobenius_norm.distance(vlmc,vlmcs[neighbor_order[i][k+1]])
            gc_distance = gc_function.distance(vlmc,vlmcs[neighbor_order[i][k+1]])
            all_distances.append((frobenius_distance, gc_distance))
    if filename is None: 
        return(all_distances)
    else:
        with open(filename,"wb") as f:
            pickle.dump(all_distances,f)



parser = argparse.ArgumentParser(description="distance parser")

add_parse_vlmc_args(parser)
add_distance_arguments(parser)

parser.add_argument(
    "-o",
    default="distances_output.pickle",
    help="output file name where the distances are saved",
)
parser.add_argument("--norm_to_gc",action='store_true',help="calculate the norm_to_gc_distance")
parser.add_argument("--number_of_neighbors",type=int, default=1,
                    help="how many neighbors should be considered when calculating norm_to_gc_distance")
parser.add_argument("--neighbor_order_file",help="file name where the neighbor order matrix is stored")

args = parser.parse_args()

vlmcs = parse_vlmcs(args, "db_config.json")
if args.norm_to_gc:
    (neighbor_order,_) = load_neighbor_order(args.neighbor_order_file)
    norm_to_gc_dist(vlmcs,neighbor_order, args.number_of_neighbors,args.o)
else:
    distance_function = parse_distance_method(args)
    neighbor_order = calculate_neighbor_order(vlmcs, distance_function)
    save_neighbor_order(neighbor_order, args.o)