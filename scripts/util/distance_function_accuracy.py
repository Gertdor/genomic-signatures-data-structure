from clustering_genomic_signatures.util.parse_vlmcs import (
    parse_vlmcs,
    add_parse_vlmc_args,
)
from clustering_genomic_signatures.util.parse_distance import (
    add_distance_arguments,
    parse_distance_method,
)

def distance_between_ids(args, pairs, names):
    args.condition='true'
    args.distance_function = 'frobenius-norm'
    frobenius_norm = parse_distance_method(args)
    vlmcs = parse_vlmcs(args, "db_config.json")
    vlmc_dict = dict([(vlmc.name,vlmc) for vlmc in vlmcs])
    distances = [frobenius_norm.distance(vlmc_dict[names[i]],vlmc_dict[names[j]]) for i,j in pairs]
    return distances


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
    while vlmc.identifier != current_distances[i]:
        closer_dists.append(current_distances[i][0])
        misses.append(current_distances[i])

    print(stats.summary(closer_dists))
    return misses
