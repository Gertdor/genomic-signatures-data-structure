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


parser = argparse.ArgumentParser(description="distance parser")

add_parse_vlmc_args(parser)
add_distance_arguments(parser)

parser.add_argument(
    "-o",
    default="all_distances.pickle",
    help="output file name where the distances are saved",
)

args = parser.parse_args()

vlmcs = parse_vlmcs(args, "db_config.json")
distance_function = parse_distance_method(args)

neighbor_order = calculate_neighbor_order(vlmcs, distance_function)
save_neighbor_order(neighbor_order, args.o)
