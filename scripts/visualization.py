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

from clustering_genomic_signatures.util.parse_distance import add_distance_arguments

from util.distance_util import distance_between_ids
from util.numberOfEqualElements import number_of_equal_elements

hyper_parameter_xlabel = (
    "hyper-parameters, P=pruning factor, K = number of neighbors, gc - gc pruning used"
)


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
    plt.title("Hyper parameter effect on number of distance calculations made")


def unpack_key(keys):
    labels = []
    for key in keys:
        labels.append(
            "GC " * key["GC"]
            + "Forest " * key["Forest"]
            + "P: "
            + str(key["p"])
            + " K: "
            + str(key["k"])
        )
    return labels


def plot_norm_to_gc(run_data):
    fig = plt.figure()
    run_data = [(r[1], r[0]) for r in run_data]
    plt.scatter(*zip(*run_data))
    plt.ylabel("Frobenius norm distance to neighbor")
    plt.xlabel("Delta GC-content")
    plt.title(
        "Frobenius distance to delta GC-content for viruses for 5 nearest neighbors"
    )


def box_plot_dist_relative(run_data):
    all_distances = run_data.get_cutoff_by_factor()
    hyperparams = run_data.get_keys()
    x_tick_labels = unpack_key(hyperparams)

    reference_distances = np.array(all_distances[0])
    delta_dist = [np.array(d) - reference_distances for d in all_distances]

    xlabel = hyper_parameter_xlabel
    ylabel = (
        "difference in distance between reference and the specific hyper parameters"
    )
    title = "effect of hyper parameters on distance to furthest NN"
    _GS_box_plot(delta_dist, x_tick_labels, xlabel, ylabel, title)


def box_plot_dist(run_data):

    all_distances = run_data.get_cutoff_by_factor()
    x_tick_labels = unpack_key(run_data.get_keys())

    xlabel = hyper_parameter_xlabel
    ylabel = "distance to nearest neighbour"
    title = "Distance to furthest NN for different hyper parameters"
    _GS_box_plot(all_distances, x_tick_labels, xlabel, ylabel, title)


def box_plot_dist_calcs(run_data):

    distance_calcs = run_data.get_ops_by_factor()
    x_tick_labels = unpack_key(run_data.get_keys())

    xlabel = hyper_parameter_xlabel
    ylabel = "number of distance calculations"
    title = "number of distance calculations made for different hyper parameters"
    _GS_box_plot(distance_calcs, x_tick_labels, xlabel, ylabel, title, True)


def _GS_box_plot(data, x_tick_labels, xlabel, ylabel, title, log=False):

    fig = plt.figure()
    border_width = 0.1
    ax_size = [
        border_width,
        2 * border_width,
        1 - 2 * border_width,
        1 - 3 * border_width,
    ]
    ax = fig.add_axes(ax_size)
    ax.boxplot(data)
    print(stats.describe(data[0]))

    xlabels = unpack_key(run_data.get_keys())
    ax.set_xticklabels(x_tick_labels)
    plt.xticks(rotation=65)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log:
        ax.set_yscale("log")

    plt.title(title)


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
    max_k = max(run_data.get_k_values())
    meta_data = get_metadata_for(names.tolist(), db_config_path)
    found_neighbors = _get_found_points(
        neighbors, run_data, include_ground_truth=True, un_pack=False
    )

    searched_points = [i for batch in signatures_used for i in batch[1]]
    number_of_searched_points = len(searched_points)

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

    genus_data_to_plot = [Counter(genus_match) for genus_match in genus_matches]
    genus_data_to_plot = [
        [counter[key] / number_of_searched_points for key in range(max_k + 1)]
        for counter in genus_data_to_plot
    ]

    family_data_to_plot = [Counter(family_match) for family_match in family_matches]
    family_data_to_plot = [
        [counter[key] / number_of_searched_points for key in range(max_k + 1)]
        for counter in family_data_to_plot
    ]

    xlabels = ["brute force"] + unpack_key(run_data.get_keys())

    # TODO seaborn nicer bars

    _bio_acc_bar_plot(
        data=family_data_to_plot,
        title="The effect of different hyper parameters on classification accuracy of family",
        x_descript=hyper_parameter_xlabel,
        y_descript="fraction of the nearest neighbors which were of the same family",
        x_tick_labels=xlabels,
        max_k=max_k,
    )

    _bio_acc_bar_plot(
        data=genus_data_to_plot,
        title="The effect of hyper parameters on classification accuracy of genus",
        x_descript=hyper_parameter_xlabel,
        y_descript="fraction nearest neighbors which were of the same genus",
        x_tick_labels=xlabels,
        max_k=max_k,
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
    bar_width = 5

    colors = plt.cm.Accent(np.linspace(0, 1, 8))

    prev = 0
    for i, genus in enumerate(data):
        for j, freq in enumerate(genus):
            ax.bar(i * (max_k + 3), freq, bottom=prev, color=colors[j], width=bar_width)
            prev = prev + freq
        prev = 0
    for i in range(max_k + 1):
        ax.bar(0, 0, color=colors[i], label=str(i))

    ax.set_title(title)
    ax.set_xlabel(x_descript)
    ax.set_ylabel(y_descript)
    ax.legend()


def _nearest_neighbor_in_all_trees(neighbor_list, signatures_used):
    NNS = []
    for (tree, points) in signatures_used:
        NNS.extend(_nearest_neighbor_in_tree(neighbor_list, tree, points))
    return NNS


# TODO add a theoretical max (when the genus/family is in the tree)
def classification_accuracy(names, run_data, db_config_path):

    found_genuses = [genuses for genuses in run_data.classify("genus")]
    found_families = [families for families in run_data.classify("family")]

    meta_data = get_metadata_for(names.tolist(), db_config_path)
    signatures_used = run_data.get_signatures_used()
    searched_points = [i for batch in signatures_used for i in batch[1]]

    true_genuses = [meta_data[point]["genus"] for point in searched_points]
    true_families = [meta_data[point]["family"] for point in searched_points]

    genus_matches = [
        sum(_number_of_equal_elements(current_genuses, true_genuses))
        / len(current_genuses)
        for current_genuses in found_genuses
    ]
    family_matches = [
        sum(_number_of_equal_elements(current_families, true_families))
        / len(current_families)
        for current_families in found_families
    ]
    values = genus_matches + family_matches
    ranks = ["genus"] * len(genus_matches) + ["family"] * len(family_matches)

    run_settings = unpack_key(run_data.get_keys()) * 2
    d = {"values": values, "ranks": ranks, "x": run_settings}
    df = pd.DataFrame(d)
    ax = sns.barplot(data=df, x="x", y="values", hue="ranks")
    ax.set_xlabel(hyper_parameter_xlabel)
    ax.set_ylabel("proportion of correctly classified queries")
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
    "--neighbor_orders", help="file for neighbor order and names for current dataset"
)

parser.add_argument(
    "--classification_accuracy",
    action="store_true",
    help="plot the classification accuracy",
)

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
    box_plot_dist_relative(run_data)
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

    with open(args.neighbor_orders, "rb") as f:
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
