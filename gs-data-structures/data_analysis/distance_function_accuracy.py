import pickle
import time
import numpy as np
from operator import itemgetter
from scipy import stats
from dataStructures.VLMCElement import VPTreeVLMC

def save_distances(distances, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(distances,f)

def load_distances(file_name):
    with open(file_name, "rb") as f:
        pickle.load(f)

def calc_pariwise_fast(vlmcs):
    i = 0
    dist=[]
    for vlmc in vlmcs:
        i+=1
        start_time=time.time()
        dist.append([vlmc.distance(other) for other in vlmcs])
        print(i, time.time()-start_time)
    return(np.array(dist))

def calculate_pairwise_distance(vlmcs):
    distances = {}
    for vlmc in vlmcs:
        
        local_dist = [(vlmc.distance(other),other.identifier) for other in vlmcs]
        local_dist.sort(key=itemgetter(0))
        distances[vlmc.identifier] = local_dist

    return(distances)


def get_distance_accuracy(distances, found_pair):
    """
    parameters
    ----------
    distances is the pairwise distance between all
    vlmc in the tree where the query was made.
    This can be calculated with:
        calculate_pariwise_distance
    from this class. Note that this must be a 
    dict where you can index with the identifier
    of the vlmc.

    found_pair is a tuple or list of length 2 where
    the first element is the vlmc queried for and
    the second element is the nearest neighbour found
    
    returns
    ------
    how many closer neighbours were found using
    the original distance function.
    """
    vlmc = found_pair[0]
    current_distances = distances[vlmc.identifier]
    i = 0
    misses = []
    closer_dists = []
    while(vlmc.identifier != current_distances[i]):
        closer_dists.append(current_distances[i][0])
        misses.append(current_distances[i])
    
    print(stats.summary(closer_dists))
    return(misses)
