import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Colormap
import argparse
import pandas as pd

from collections import Counter
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
    k_values = run_data.get_k_values()

    fig = plt.figure()
    border_width = 0.1
    ax_size = [
        border_width,
        2 * border_width,
        1 - (2 * border_width),
        1 - (3 * border_width),
    ]
    ax = fig.add_axes(ax_size)
    ax.boxplot(distances)
    pruning_factors = [str(round(x, 2)) for x in greedy_factors]
    xlabels = ["P:" + p + "  K:" + str(k) for (p, k) in zip(pruning_factors, k_values)]
    ax.set_xticklabels(xlabels)
    plt.xticks(rotation=65)
    plt.xlabel("pruning factor (P) and number of neighbors (K)")
    plt.ylabel("distance to nearest neighbour")
    plt.title("Pruning effect on distance to nearest neighbour")


def box_plot_dist_calcs(run_data):

    distances = run_data.get_ops_by_factor()
    greedy_factors = run_data.get_greedy_factors()
    k_values = run_data.get_k_values()

    fig = plt.figure()
    border_width = 0.1
    ax_size = [0.1, 0 + 2 * border_width, 1 - 2 * border_width, 1 - 3 * border_width]
    ax = fig.add_axes(ax_size)
    ax.boxplot(distances)
    print(stats.describe(distances[0]))

    pruning_factors = [str(round(x, 2)) for x in greedy_factors]
    xlabels = ["P:" + p + "  K:" + str(k) for (p, k) in zip(pruning_factors, k_values)]
    ax.set_xticklabels(xlabels)
    plt.xticks(rotation=65)
    plt.xlabel("pruning factor (P) and number of neighbors (K)")
    plt.ylabel("number of distance calculations")
    ax.set_yscale("log")

    plt.title("Pruning effect on the number of distance calculations made")


def _get_found_points(neighbors, run_data, include_ground_truth=True, un_pack=True):

    found_neighbors = run_data.get_ids_by_factor(un_pack)

    if include_ground_truth:
        signatures_used = run_data.get_signatures_used()
        if un_pack:
            true_nns = _nearest_neighbor_in_all_trees(neighbors, signatures_used)
        else:
            true_nns = [
                [nn]
                for nn in _nearest_neighbor_in_all_trees(neighbors, signatures_used)
            ]

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
            dist = 0
            current_genus = meta_data[names[point]]["genus"]
            current_family = meta_data[names[point]]["family"]
            current_neighbor = neighbors[point][i]
            while (
                (
                    current_genus != meta_data[names[current_neighbor]]["genus"]
                    or current_neighbor not in tree
                    or current_neighbor == point
                )
                and dist < max_distance
                and i < len(neighbors[point]) - 1
            ):
                if current_neighbor in tree:
                    dist += 1
                i += 1
                current_neighbor = neighbors[point][i]
            genus_distances.append((dist, (point, neighbors[point][i])))
            i = 0
            dist = 0
            current_neighbor = neighbors[point][i]
            while (
                current_family != meta_data[names[current_neighbor]]["family"]
                or current_neighbor not in tree
                or current_neighbor == point
            ) and (dist < max_distance and i < len(neighbors[point]) - 1):
                if current_neighbor in tree:
                    dist += 1
                i += 1
                current_neighbor = neighbors[point][i]
            family_distances.append((dist, (point, neighbors[point][i])))

    return (genus_distances, family_distances)


def plot_signature_dist_to_match(neighbors, names, run_data, max_distance=50):

    genus_distances, family_distances = _true_distance_to_bio_match(
        neighbors, names, run_data, max_distance
    )

    genus_distances = [g[0] for g in genus_distances]
    family_distances = [f[0] for f in family_distances]
    plt.figure()
    plt.hist(genus_distances, max_distance, facecolor="blue")
    plt.xlabel("number of non matching closer neighbors")
    plt.ylabel("number of occurances")
    plt.title("distance in signatures to closest signature of the same genus")
    plt.figure()
    plt.xlabel("number of non matching closer neighbors")
    plt.ylabel("number of occurances")
    plt.title("distance in signatures to closest signature of the same family")
    plt.hist(family_distances, max_distance, facecolor="green")


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


# TODO maybe a bug, look over this?
def exact_matches(neighbors, names, run_data):
    all_points = _get_found_points(neighbors, run_data, include_ground_truth=True)
    ground_truth = all_points[0]
    rest = all_points[1:]
    number_of_matches = [
        sum(_number_of_equal_elements(ground_truth, current)) for current in rest
    ]

    print("number of matches: ", number_of_matches)
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(range(len(number_of_matches)), number_of_matches)
    plt.ylim(0, max(number_of_matches))


def biological_accuracy(neighbors, names, run_data, db_config_path):

    signatures_used = run_data.get_signatures_used()
    greedy_factors = run_data.get_greedy_factors()
    k_values = run_data.get_k_values()
    meta_data = get_metadata_for(names.tolist(), db_config_path)
    found_neighbors = _get_found_points(
        neighbors, run_data, include_ground_truth=True, un_pack=False
    )

    searched_points = [i for batch in signatures_used for i in batch[1]]
    genuses = [meta_data[names[point]]["genus"] for point in searched_points]
    families = [meta_data[names[point]]["family"] for point in searched_points]

    found_genuses = [
        [[meta_data[names[nn]]["genus"] for nn in NNS] for NNS in found_neighbor]
        for found_neighbor in found_neighbors
    ]
    found_families = [
        [[meta_data[names[nn]]["family"] for nn in NNS] for NNS in found_neighbor]
        for found_neighbor in found_neighbors
    ]

    genus_matches = [
        [NNS.count(query) for query, NNS in zip(genuses, run)] for run in found_genuses
    ]

    family_matches = [
        [NNS.count(query) for query, NNS in zip(families, run)]
        for run in found_families
    ]

    max_k = max(k_values)
    genus_data_to_plot = [Counter(genus_match) for genus_match in genus_matches]
    genus_data_to_plot = [
        [counter[key] for key in range(max_k + 1)] for counter in genus_data_to_plot
    ]

    family_data_to_plot = [Counter(family_match) for family_match in family_matches]
    family_data_to_plot = [
        [counter[key] for key in range(max_k + 1)] for counter in family_data_to_plot
    ]
    k_values = run_data.get_k_values()

    xlabels = ["brute force"] + [
        "P:" + str(round(p, 2)) + "  K:" + str(round(k, 2))
        for (p, k) in zip(greedy_factors, k_values)
    ]
    # TODO seaborn nicer bars

    _bio_acc_bar_plot(
        data=family_data_to_plot,
        title="The effect of pruning factor on classification accuracy of family",
        x_descript="hyper-parameters, P=pruning factor, K = number of neighbors",
        y_descript="frequency at which exactly [color] of the nearest neighbors were of the same family",
        x_tick_labels=xlabels,
        max_k=max(k_values),
    )

    _bio_acc_bar_plot(
        data=genus_data_to_plot,
        title="The effect of pruning factor on classification accuracy of genus",
        x_descript="hyper-parameters, P=pruning factor, K = number of neighbors",
        y_descript="frequency at which exactly [color] of the nearest neighbors were of the same genus",
        x_tick_labels=xlabels,
        max_k=max(k_values),
    )


def _bio_acc_bar_plot(data, title, x_descript, y_descript, x_tick_labels, max_k):

    fig = plt.figure()
    border_width = 0.1
    ax_size = [
        border_width,
        2 * border_width,
        1 - (2 * border_width),
        1 - (3 * border_width),
    ]
    ax = fig.add_axes(ax_size)

    xlabel_locs = np.arange(0, len(x_tick_labels) * (max_k + 3), step=(max_k + 3))
    plt.xticks(xlabel_locs, x_tick_labels, rotation=65)
    bar_width = 0.8

    colors = plt.cm.Accent(np.linspace(0, 1, 8))
    for i, genus in enumerate(data):
        for j, freq in enumerate(genus):
            ax.bar(i * (max_k + 3) + j + 1, freq, color=colors[j])

    for i in range(max_k + 1):
        ax.bar(0, 0, color=colors[i], label=str(i))

    ax.set_title(title)
    ax.set_xlabel(x_descript)
    ax.set_ylabel(y_descript)
    ax.legend()


def _number_of_equal_elements(list1, list2):
    return [x == y for x, y in zip(list1, list2)]


def _nearest_neighbor_in_all_trees(neighbor_list, signatures_used):
    NNS = []
    for (tree, points) in signatures_used:
        NNS.extend(_nearest_neighbor_in_tree(neighbor_list, tree, points))
    return NNS


# TODO add a theoretical max (when the genus/family is in the tree)
def classification_accuracy(names, run_data, db_config_path):

    found_genuses = [[g[0] for g in genuses] for genuses in run_data.classify("genus")]
    found_families = [
        [f[0] for f in families] for families in run_data.classify("family")
    ]

    meta_data = get_metadata_for(names.tolist(), db_config_path)
    signatures_used = run_data.get_signatures_used()
    searched_points = [i for batch in signatures_used for i in batch[1]]
    true_genuses = [meta_data[names[point]]["genus"] for point in searched_points]
    true_families = [meta_data[names[point]]["family"] for point in searched_points]

    genus_matches = [
        sum(_number_of_equal_elements(current_genuses, true_genuses))
        for current_genuses in found_genuses
    ]
    family_matches = [
        sum(_number_of_equal_elements(current_families, true_families))
        for current_families in found_families
    ]
    values = genus_matches + family_matches
    ranks = ["genus"] * len(genus_matches) + ["family"] * len(family_matches)

    run_settings = run_data.get_keys() * 2
    print(run_settings)
    d = {"values": values, "ranks": ranks, "x": run_settings}
    df = pd.DataFrame(d)
    ax = sns.barplot(data=df, x="x", y="values", hue="ranks")
    ax.set_xlabel(
        "Settings for the run. P = greedy pruning factor, K = number of neighbors"
    )
    ax.set_ylabel("number of correctly classified queries")
    ax.set_title("Classification accuracy")


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

parser.add_argument(
    "--classification_accuracy",
    action="store_true",
    help="plot the classification accuracy",
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
    or args.classification_accuracy
):

    with open(args.distance_file, "rb") as f:
        (neighbors, names) = pickle.load(f)
    if args.classification_accuracy:
        classification_accuracy(names, run_data, db_config_path)

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
