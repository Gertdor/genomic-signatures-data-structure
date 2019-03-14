import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from clustering_genomic_signatures.dbtools.get_signature_metadata import get_metadata_for

def plot_dist_calc_to_distance(filename, variance=False):
    with open(filename,"rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)

    all_stats=[]    
    for factor in all_runs:
        run = [nn for tree in factor for nn in tree]
        run_stats = stats.describe(run)
        if(variance):
            all_stats.append((run_stats.skewness[1], run_stats.mean[2]))
        else:
            all_stats.append(run_stats.mean)

    distance = [stat[1] for stat in all_stats]
    dist_calcs = [stat[2] for stat in all_stats]
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(distance,dist_calcs)
    plt.xlabel("average distance to nearest neighbor")
    plt.ylabel("average number of distance calculations")
    plt.title("Pruning effect on number of distance calculations made")

def box_plot_dist(filename):
    with open(filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    distances = [[nn[1] for tree in factor for nn in tree] for factor in all_runs]
    
    xlabels = [round(x,2) for x in greedy_factors]
    ax.boxplot(distances)
    ax.set_xticklabels(xlabels)
    #ax.hist(distances[9])
    #ax.hist(distances[0])
    plt.xlabel("pruning factor")
    plt.ylabel("distance to nearest neighbour")
    plt.title("Pruning effect on distance to nearest neighbour")

def box_plot_dist_calcs(filename):
    with open(filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    distances = [[nn[2] for tree in factor for nn in tree] for factor in all_runs]
    
    xlabels = [round(x,2) for x in greedy_factors]
    ax.boxplot(distances)
    ax.set_xticklabels(xlabels)
    plt.xlabel("pruning factor")
    plt.ylabel("number of distance calculations")
    plt.title("Pruning effect on the number of distance calculations made")

def biological_accuracy(run_filename, neighbors_filename, db_config_path):
    with open(run_filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)
    with open(neighbors_filename, "rb") as f:
        (neighbors,names) = pickle.load(f)
    
    meta_data = get_metadata_for(names.tolist(),db_config_path)
    
    found_neighbors = [[nn[0] for tree in factor for nn in tree] for factor in all_runs]
    searched_points = [i for batch in signatures_used for i in batch[1]]
    
    print(len(found_neighbors[0]))
    print(len(searched_points))
    
    genuses = [meta_data[names[point]]["genus"] for point in searched_points]
    families = [meta_data[names[point]]["family"] for point in searched_points]
    print(genuses[0:5])
    print(families[0:5])
    found_genuses = [[meta_data[names[nn]]["genus"] for nn in found_neighbor] for found_neighbor in found_neighbors]
    found_families = [[meta_data[names[nn]]["family"] for nn in found_neighbor] for found_neighbor in found_neighbors] 
    print(found_genuses[0][0:5])
    print(found_families[0][0:5])
    genus_matches = [number_of_equal_elements(genuses, genuses_current_try) for genuses_current_try in found_genuses]
    family_matches = [number_of_equal_elements(families, families_current_try) for families_current_try in found_families]

    print("genus matches", genus_matches)
    print("family_matches", family_matches)


def number_of_equal_elements(list1,list2):
    return sum([x==y for x,y in zip(list1,list2)])


def get_prediction_accuracy(run_filename, neighbors_filename, dbconfig_path):
    with open(run_filename, "rb") as f:
        (all_runs, greedy_factors, signatures_used) = pickle.load(f)
    with open(neighbors_filename, "rb") as f:
        (neighbors,names) = pickle.load(f)
    meta_data = get_metadata_for(names,dbconfig_path)
    

db_config_path = "db_config.json"
fileName = "10run400searchFullWithNNInfo.pickle"
neighbors_filename = "all_neighbors.pickle"
#plt.style.use('seaborn-whitegrid')
#plot_dist_calc_to_distance(fileName, False)
#plt.xlim(0.,0.04)
biological_accuracy(fileName, neighbors_filename, db_config_path)
#box_plot_dist(fileName)
#box_plot_dist_calcs(fileName)

#plt.show()
