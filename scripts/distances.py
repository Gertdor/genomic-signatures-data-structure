import argparse
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
import time
import pandas as pd

from operator import itemgetter
from scipy import stats
from dataStructures.VLMCElement import VPTreeVLMC
from operator import itemgetter

from clustering_genomic_signatures.util.parse_signatures import (
    parse_signatures,
    add_parse_signature_args,
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
    distances = np.array(distance_function.distances(vlmcs))
    neighbor_order = np.argsort(distances, 1)
    return (neighbor_order, names, distances)


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
    args.distance_function = "frobenius-norm"
    frobenius_norm = parse_distance_method(args)
    args.distance_function = "gc-content"
    gc_function = parse_distance_method(args)

    all_distances = []
    for i, vlmc in enumerate(vlmcs):
        for k in range(number_of_neighbors):
            frobenius_distance = frobenius_norm.distance(
                vlmc, vlmcs[neighbor_order[i][k + 1]]
            )
            gc_distance = gc_function.distance(vlmc, vlmcs[neighbor_order[i][k + 1]])
            all_distances.append((frobenius_distance, gc_distance))
    if filename is None:
        return all_distances
    else:
        with open(filename, "wb") as f:
            pickle.dump(all_distances, f)


def gc_dist_distribution(vlmcs):
    gc = [vlmc.tree[""]["G"] + vlmc.tree[""]["C"] for vlmc in vlmcs]
    ax = sns.distplot(gc, norm_hist=True)
    plt.xlabel("gc content")
    plt.ylabel("density")
    plt.title("distribution of gc content in the database")
    plt.show()


def branch_length_distribution(vlmcs):
    length_list = [[len(key) for key in vlmc.tree.keys()] for vlmc in vlmcs]
    max_depths = [max(lengths) for lengths in length_list]
    numberOfParam = [len(lengths) for lengths in length_list]
    ys = [
        sum([length > max_depth / 2 for length in lengths])
        for max_depth, lengths in zip(max_depths, length_list)
    ]
    plt.subplots()
    plt.boxplot(numberOfParam)
    plt.subplots()
    d = {"max_depth": max_depths, "ys": ys}
    df = pd.DataFrame(data=d)
    sns.catplot(x="max_depth", y="ys", data=df)
    plt.xlabel("max depth of VLMC")
    plt.ylabel("number of branches with depth > maxdepth/2")
    plt.title("distribution of branch depths compared to max depth")
    plt.show()


parser = argparse.ArgumentParser(description="distance parser")

add_parse_signature_args(parser)
add_distance_arguments(parser)

parser.add_argument(
    "-o",
    default="neighbor_order.pickle",
    help="output file name where the beighbor orders are saved",
)
parser.add_argument(
    "--distance_file",
    default="distances.pickle",
    help="name of file to save distances in",
)
parser.add_argument(
    "--norm_to_gc", action="store_true", help="calculate the norm_to_gc_distance"
)
parser.add_argument(
    "--number_of_neighbors",
    type=int,
    default=1,
    help="how many neighbors should be considered when calculating norm_to_gc_distance",
)
parser.add_argument(
    "--gc_content", action="store_true", help="calculate gc content of all signatures"
)

parser.add_argument(
    "--neighbor_order",
    action="store_true",
    help="calculate distance and neighbor order",
)

args = parser.parse_args()

vlmcs = parse_signatures(args, "../settings/db_config.json")

print("length", len(vlmcs))

# branch_length_distribution(vlmcs)

if args.gc_content:
    gc_dist_distribution(vlmcs)

if args.norm_to_gc:
    (neighbor_order, _) = load_neighbor_order(args.neighbor_order_file)
    norm_to_gc_dist(vlmcs, neighbor_order, args.number_of_neighbors, args.o)
if args.neighbor_order:
    start = time.time()
    distance_function = parse_distance_method(args)
    (neighbor_order, names, distances) = calculate_neighbor_order(
        vlmcs, distance_function
    )
    save_neighbor_order((neighbor_order, names), args.o)
    with open(args.distance_file, "wb") as f:
        pickle.dump(distances, f)
    print(time.time() - start)
