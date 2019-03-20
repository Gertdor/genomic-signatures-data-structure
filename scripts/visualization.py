import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy import stats
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)
from clustering_genomic_signatures.util.parse_vlmcs import add_parse_vlmc_args
from clustering_genomic_signatures.util.parse_distance import add_distance_arguments

from util.distance_util import distance_between_ids


def plot_dist_calc_to_distance(run_data, variance=False):

    all_distances = run_data.get_distances_by_factor()
    all_ops = run_data.get_ops_by_factor(repeat_k=True)
    all_runs = [zip(a, b) for a, b in zip(all_distances, all_ops)]
    all_runs = [[r for r in run] for run in all_runs]
    all_stats = []
    for run in all_runs:
        run_stats = stats.describe(run)
        if variance:
            all_stats.append((run_stats.skewness[1], run_stats.mean[2]))
        else:
            all_stats.append(run_stats.mean)

    distance = [stat[0] for stat in all_stats]
    dist_calcs = [stat[1] for stat in all_stats]
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(distance, dist_calcs)
    plt.xlabel("average distance to nearest neighbor")
    plt.ylabel("average number of distance calculations")
    plt.title("Pruning effect on number of distance calculations made")


def plot_norm_to_gc(run_data):
    fig = plt.figure()
    run_data = [(r[1], r[0]) for r in run_data]
    plt.scatter(*zip(*run_data))
    plt.ylabel("Frobenius norm distance to neighbor")
    plt.xlabel("Delta GC-content")
    plt.title(
        "Frobenius distance to delta GC-content for viruses for 5 nearest neighbors"
    )


def box_plot_dist(run_data):

    distances = run_data.get_distances_by_factor()
    greedy_factors = run_data.get_greedy_factors()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xlabels = [round(x, 2) for x in greedy_factors]
    ax.boxplot(distances)
    ax.set_xticklabels(xlabels)
    plt.xlabel("pruning factor")
    plt.ylabel("distance to nearest neighbour")
    plt.title("Pruning effect on distance to nearest neighbour")


def box_plot_dist_calcs(run_data):

    distances = run_data.get_distances_by_factor()
    greedy_factors = run_data.get_greedy_factors()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xlabels = [round(x, 2) for x in greedy_factors]
    ax.boxplot(distances)
    ax.set_xticklabels(xlabels)
    plt.xlabel("pruning factor")
    plt.ylabel("number of distance calculations")
    ax.set_yscale("log")
    plt.title("Pruning effect on the number of distance calculations made")


def _get_found_points(neighbors, run_data, include_ground_truth=True):

    found_neighbors = run_data.get_ids_by_factor()
    signatures_used = run_data.get_signatures_used()

    if include_ground_truth:
        true_nns = _nearest_neighbor_in_all_trees(neighbors, signatures_used)
        found_neighbors.insert(0, true_nns)

    return found_neighbors


def _true_distance_to_bio_match(neighbors, names, run_data, max_distance=50):

    signatures_used = run_data.get_signatures_used()
    meta_data = get_metadata_for(names.tolist(), db_config_path)

    genus_distances = []
    family_distances = []
    for tree, point_set in signatures_used:
        for point in point_set:
            i = 0
            current_genus = meta_data[names[point]]["genus"]
            current_family = meta_data[names[point]]["family"]
            current_neighbor = neighbors[point][i]
            while (
                current_genus != meta_data[names[current_neighbor]]["genus"]
                or current_neighbor not in tree
                or current_neighbor == point
            ) and i < max_distance:
                i += 1
                current_neighbor = neighbors[point][i]
            genus_distances.append((i, (point, neighbors[point][i])))
            i = 0
            current_neighbor = neighbors[point][i]
            while (
                current_family != meta_data[names[current_neighbor]]["family"]
                or current_neighbor not in tree
                or current_neighbor == point
            ) and i < max_distance:
                i += 1
                current_neighbor = neighbors[point][i]
            family_distances.append((i, (point, neighbors[point][i])))

    return (genus_distances, family_distances)


def plot_signature_dist_to_match(neighbors, names, run_data, max_distance=50):

    genus_distances, family_distances = _true_distance_to_bio_match(
        neighbors, names, run_data, max_distance
    )

    genus_distances = [g[0] for g in genus_distances]
    family_distances = [f[0] for f in family_distances]
    plt.figure()
    plt.hist(genus_distances, 50, facecolor="blue")
    plt.xlabel("number of non matching closer neighbors")
    plt.ylabel("number of occurances")
    plt.title("distance in signatures to closest signature of the same genus")
    plt.figure()
    plt.xlabel("number of non matching closer neighbors")
    plt.ylabel("number of occurances")
    plt.title("distance in signatures to closest signature of the same family")
    plt.hist(family_distances, 50, facecolor="green")


def plot_FN_dist_to_match(args, neighbors, names, run_data, max_signature_distance=50):

    genus_distances, family_distances = _true_distance_to_bio_match(
        neighbors, names, run_data, max_signature_distance
    )

    genus_pairs = [g[1] for g in genus_distances if g[0] < max_signature_distance - 1]
    family_pairs = [f[1] for f in family_distances if f[0] < max_signature_distance - 1]
    genus_distances = distance_between_ids(args, genus_pairs, names)
    family_distances = distance_between_ids(args, family_pairs, names)

    plt.figure()
    plt.hist(genus_distances, 50, facecolor="blue")
    plt.xlabel("Frobenius norm distance to closest matching neighbor")
    plt.ylabel("number of occurances")
    plt.title("frobenius norm distance to closest signature of the same genus")
    plt.figure()
    plt.xlabel("frobenius distance to closest matching neighbor")
    plt.ylabel("number of occurances")
    plt.title("frobenius norm distance to closest signature of the same family")
    plt.hist(family_distances, 50, facecolor="green")


def exact_matches(neighbors, names, run_data):
    all_points = _get_found_points(neighbors, run_data, True)
    ground_truth = all_points[0]
    rest = all_points[1:]
    print(len(rest))
    number_of_matches = [
        _number_of_equal_elements(ground_truth, current) for current in rest
    ]

    print(number_of_matches)
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(range(len(number_of_matches)), number_of_matches)
    plt.ylim(0, max(number_of_matches))


def biological_accuracy(neighbors, names, run_data, db_config_path):

    signatures_used = run_data.get_signatures_used()
    greedy_factors = run_data.get_greedy_factors()

    meta_data = get_metadata_for(names.tolist(), db_config_path)
    found_neighbors = _get_found_points(neighbors, run_data)

    searched_points = [i for batch in signatures_used for i in batch[1]]
    genuses = [meta_data[names[point]]["genus"] for point in searched_points]
    families = [meta_data[names[point]]["family"] for point in searched_points]

    found_genuses = [
        [meta_data[names[nn]]["genus"] for nn in found_neighbor]
        for found_neighbor in found_neighbors
    ]
    found_families = [
        [meta_data[names[nn]]["family"] for nn in found_neighbor]
        for found_neighbor in found_neighbors
    ]

    genus_matches = [
        _number_of_equal_elements(genuses, genuses_current_try)
        for genuses_current_try in found_genuses
    ]
    family_matches = [
        _number_of_equal_elements(families, families_current_try)
        for families_current_try in found_families
    ]

    genus_matches = np.array(genus_matches) / len(searched_points)
    family_matches = np.array(family_matches) / len(searched_points)

    # TODO seaborn nicer bars

    fig, ax = plt.subplots()
    xlabels = ["brute force"] + [str(round(f, 1)) for f in greedy_factors]
    bar_width = 0.6
    bar_loc = np.arange(len(xlabels)) * 1.5
    bars1 = ax.bar(bar_loc, genus_matches, width=bar_width, color="g")
    bars2 = ax.bar(bar_loc + bar_width, family_matches, width=bar_width, color="b")

    ax.set_title("The effect of pruning factor on classification accuracy")
    ax.set_xlabel("pruning factor")
    ax.set_ylabel("prediction accuracy")
    ax.set_xticks(bar_loc + bar_width / 2)
    ax.set_xticklabels(xlabels)
    ax.legend((bars1[0], bars2[0]), ("genus", "family"))

    print("genus matches", genus_matches)
    print("family_matches", family_matches)


def _number_of_equal_elements(list1, list2):
    return sum([x == y for x, y in zip(list1, list2)])


def _nearest_neighbor_in_all_trees(neighbor_list, signatures_used):
    NNS = []
    for (tree, points) in signatures_used:
        NNS.extend(_nearest_neighbor_in_tree(neighbor_list, tree, points))
    return NNS


def _nearest_neighbor_in_tree(neighbor_array, tree, points):
    number_of_elements = len(neighbor_array[0])
    elements_in_tree = np.zeros(number_of_elements)
    neighbors = []

    for elem in tree:  # Want the ones not in the tree still to be 0
        elements_in_tree[elem] = 1

    for point in points:
        i = 0
        while not elements_in_tree[neighbor_array[point][i]]:
            i += 1
        neighbors.append(neighbor_array[point][i])
    return neighbors


parser = argparse.ArgumentParser(description="visualization script args")

parser.add_argument(
    "--boxplots", action="store_true", help="should boxplots be generated"
)
parser.add_argument(
    "--bio_accuracy",
    action="store_true",
    help="should the biological accuracy be calculated?",
)
parser.add_argument(
    "--exact_matches",
    action="store_true",
    help="should the number of exact matches be plotted",
)
parser.add_argument(
    "--acc_to_dist",
    action="store_true",
    help="should the accuracy to number of distance calculations be plotted?",
)
parser.add_argument(
    "--dist_to_match",
    action="store_true",
    help="calculate the distance to the first genus/family match",
)
parser.add_argument(
    "--norm_to_gc", action="store_true", help="plot NN distance to gc distance"
)
parser.add_argument(
    "--test_all",
    action="store_true",
    help="Should all functions be tested with a small test set",
)

parser.add_argument(
    "--fn_dist_to_match",
    action="store_true",
    help="calculate the frobenius norm distance to the nearest signature of the same genus/family",
)

parser.add_argument("--input_file", help="file name of greedy run data")
parser.add_argument(
    "--distance_file", help="file for pairwise distance and names for current dataset"
)

add_parse_vlmc_args(parser)
add_distance_arguments(parser)

args = parser.parse_args()

db_config_path = "db_config.json"

if args.test_all:
    with open("data/test_sets/virus_gc_norm.pickle", "rb") as f:
        run_data = pickle.load(f)

    plot_norm_to_gc(run_data)

    with open("data/test_sets/greedy_factor_test.pickle", "rb") as f:
        run_data = pickle.load(f)

    with open("data/test_sets/small_distance.pickle", "rb") as f:
        (neighbors, names) = pickle.load(f)

    box_plot_dist(run_data)
    box_plot_dist_calcs(run_data)
    plot_dist_calc_to_distance(run_data, False)
    plot_FN_dist_to_match(args, neighbors, names, run_data)
    plot_signature_dist_to_match(neighbors, names, run_data)
    exact_matches(neighbors, names, run_data)
    biological_accuracy(neighbors, names, run_data, db_config_path)
    plt.show()
    exit()

with open(args.input_file, "rb") as f:
    run_data = pickle.load(f)

if args.norm_to_gc:
    plot_norm_to_gc(run_data)

if args.boxplots:
    box_plot_dist(run_data)
    box_plot_dist_calcs(run_data)

if args.acc_to_dist:
    plot_dist_calc_to_distance(run_data, False)

if (
    args.bio_accuracy
    or args.dist_to_match
    or args.exact_matches
    or args.fn_dist_to_match
):

    with open(args.distance_file, "rb") as f:
        (neighbors, names) = pickle.load(f)

    if args.fn_dist_to_match:
        plot_FN_dist_to_match(args, neighbors, names, run_data)

    if args.exact_matches:
        exact_matches(neighbors, names, run_data)

    if args.bio_accuracy:
        biological_accuracy(neighbors, names, run_data, db_config_path)

    if args.dist_to_match:
        plot_signature_dist_to_match(neighbors, names, run_data)

plt.style.use("seaborn-whitegrid")
plt.show()
