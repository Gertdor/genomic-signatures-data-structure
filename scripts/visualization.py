import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy import stats
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)


def plot_dist_calc_to_distance(filename, variance=False):
    with open(filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)

    all_stats = []
    for factor in all_runs:
        run = [nn for tree in factor for nn in tree]
        run_stats = stats.describe(run)
        if variance:
            all_stats.append((run_stats.skewness[1], run_stats.mean[2]))
        else:
            all_stats.append(run_stats.mean)

    distance = [stat[1] for stat in all_stats]
    dist_calcs = [stat[2] for stat in all_stats]
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(distance, dist_calcs)
    plt.xlabel("average distance to nearest neighbor")
    plt.ylabel("average number of distance calculations")
    plt.title("Pruning effect on number of distance calculations made")


def box_plot_dist(filename):
    with open(filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    distances = [[nn[1] for tree in factor for nn in tree] for factor in all_runs]

    xlabels = [round(x, 2) for x in greedy_factors]
    ax.boxplot(distances)
    ax.set_xticklabels(xlabels)
    plt.xlabel("pruning factor")
    plt.ylabel("distance to nearest neighbour")
    plt.title("Pruning effect on distance to nearest neighbour")


def box_plot_dist_calcs(filename):
    with open(filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    distances = [[nn[2] for tree in factor for nn in tree] for factor in all_runs]

    xlabels = [round(x, 2) for x in greedy_factors]
    ax.boxplot(distances)
    ax.set_xticklabels(xlabels)
    plt.xlabel("pruning factor")
    plt.ylabel("number of distance calculations")
    ax.set_yscale("log")
    plt.title("Pruning effect on the number of distance calculations made")


def get_found_points(run_filename, neighbors_filename, include_ground_truth=True):

    with open(run_filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)

    found_neighbors = [[nn[0] for tree in factor for nn in tree] for factor in all_runs]

    if include_ground_truth:
        with open(neighbors_filename, "rb") as f:
            (neighbors, names) = pickle.load(f)

        true_nns = nearest_neighbor_in_all_trees(neighbors, signatures_used)
        found_neighbors.insert(0, true_nns)

    return found_neighbors


def exact_matches(run_filename, neighbors_filename):
    all_points = get_found_points(run_filename, neighbors_filename, True)
    ground_truth = all_points[0]
    rest = all_points[1:]
    print(len(rest))
    number_of_matches = [
        number_of_equal_elements(ground_truth, current) for current in rest
    ]

    print(number_of_matches)
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(range(len(number_of_matches)), number_of_matches)
    plt.ylim(0, max(number_of_matches))


def biological_accuracy(run_filename, neighbors_filename, db_config_path):

    meta_data = get_metadata_for(names.tolist(), db_config_path)
    found_neighbors = get_found_points(run_filename, neighbors_filename)

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
        number_of_equal_elements(genuses, genuses_current_try)
        for genuses_current_try in found_genuses
    ]
    family_matches = [
        number_of_equal_elements(families, families_current_try)
        for families_current_try in found_families
    ]
    print("genus matches", genus_matches)
    print("family_matches", family_matches)


def number_of_equal_elements(list1, list2):
    return sum([x == y for x, y in zip(list1, list2)])


def nearest_neighbor_in_all_trees(neighbor_list, signatures_used):
    NNS = []
    for (tree, points) in signatures_used:
        NNS.extend(nearest_neighbor_in_tree(neighbor_list, tree, points))
    return NNS


def nearest_neighbor_in_tree(neighbor_array, tree, points):
    number_of_elements = len(neighbor_array[0])
    elements_in_tree = np.zeros(number_of_elements)
    neighbors = []

    for elem in tree:
        elements_in_tree[elem] = 1
    for point in points:
        i = 0
        while not elements_in_tree[neighbor_array[point][i]]:
            i += 1
        neighbors.append(neighbor_array[point][i])
    return neighbors


def get_prediction_accuracy(run_filename, neighbors_filename, dbconfig_path):
    with open(run_filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)
    with open(neighbors_filename, "rb") as f:
        (neighbors, names) = pickle.load(f)
    meta_data = get_metadata_for(names, dbconfig_path)


parser = argparse.ArgumentParser(description="visualization script args")

db_config_path = "db_config.json"
fileName = "10run400searchFullWithNNInfo.pickle"
# fileName = "greedy_factor_test.pickle"
neighbors_filename = "all_neighbors.pickle"
exact_matches(fileName, neighbors_filename)

# plt.style.use('seaborn-whitegrid')
# plot_dist_calc_to_distance(fileName, False)
# biological_accuracy(fileName, neighbors_filename, db_config_path)
# box_plot_dist(fileName)
# box_plot_dist_calcs(fileName)

plt.show()
