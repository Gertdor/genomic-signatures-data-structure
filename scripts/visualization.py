import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

def plot_dist_calc_to_distance(filename, variance=False):
    with open(filename,"rb") as f:
        (all_runs, greedy_factor) = pickle.load(f)
        
    for run in all_runs:
        run_stats = stats.describe(run)
        print(run_stats)
        if(variance):
            all_stats.append((run_stats.skewness[0], run_stats.mean[1]))
        else:
            all_stats.append(run_stats.mean)

    distance = [stat[0] for stat in all_stats]
    dist_calcs = [stat[1] for stat in all_stats]
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(distance,dist_calcs)

def box_plot_dist(filename):
    with open(filename, "rb") as f:
        (all_runs, greedy_factors) = pickle.load(f)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    distances = [[r[0] for r in run] for run in all_runs]
    
    xlabels = [round(x,2) for x in greedy_factors]
    ax.boxplot(distances)
    ax.set_xticklabels(xlabels)
    plt.xlabel("pruning factor")
    plt.ylabel("distance to nearest neighbour")
    plt.title("Pruning effect on distance to nearest neighbour")

def box_plot_dist_calcs(filename):
    with open(filename, "rb") as f:
        (all_runs, greedy_factors) = pickle.load(f)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    distances = [[r[1] for r in run] for run in all_runs]
    
    xlabels = [round(x,2) for x in greedy_factors]
    ax.boxplot(distances)
    ax.set_xticklabels(xlabels)
    plt.xlabel("pruning factor")
    plt.ylabel("number of distance calculations")
    plt.title("Pruning effect on the number of distance calculations made")
def plot_speedup_vs_distance(filename):
    (all_runs, greedy_factor) = pickle.load(filename)


fileName = "10run100searchFull.pickle"
#plt.style.use('seaborn-whitegrid')
#plot_dist_calc_to_distance(fileName, False)
#plt.xlim(0.032,0.038)

box_plot_dist(fileName)
box_plot_dist_calcs(fileName)

plt.show()
