def get_distance_accuracy(distances, found_pair):
    """
    parameters
    ----------
    distances is the pairwise distance between all
    vlmc in the tree where the query was made.
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
