import argparse
import pickle
import numpy as np
from operator import itemgetter
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)
from util.numberOfEqualElements import number_of_equal_elements
from collections import defaultdict

class neighborMatrix:
    
    def __init__(self, neighbor_order, names, distances, k):
        NNS = neighbor_order[:, 1:k+1]
        NNS_dist = [[distances[i,name] for name in names] for i,names in enumerate(NNS)]

        self.NNS = {
            name:[names[n] for n in neighbors] for name, neighbors in zip(names, NNS)
        }
        self.distances = {
            name:dist for name, dist in zip(names, NNS_dist)
        }
        self.fails=0

    def classify_all(self, points, meta_data, true):
        classes = [self._classify_one(point, meta_data, t) for point,t in zip(points,true)]
        return zip(*classes)

    def _classify_one(self, point, meta_data, true):
        genuses = [meta_data[neighbor]["genus"] for neighbor in self.NNS[point]]
        families = [meta_data[neighbor]["family"] for neighbor in self.NNS[point]]
        genus_count = {name:0 for name in genuses}
        family_count = {name:0 for name in families}
        for genus,family,dist in zip (genuses, families, self.distances[point]):
            genus_count[genus] = genus_count[genus] + 1/(dist+1e-30)
            family_count[family] = family_count[family] + 1/(dist+1e-30)
        genus = max(genus_count.items(),key=itemgetter(1))[0]
        family = max(family_count.items(),key=itemgetter(1))[0]
        return(genus,family)

def default_class():
    print("fail")
    return {"genus":"none","family":"none"}

def classify_test(original, other, meta_data, name_intersect):
    
    true_genuses = [meta_data[point]["genus"] for point in name_intersect]
    true_families = [meta_data[point]["family"] for point in name_intersect]
    (orig_genuses, orig_families) = original.classify_all(name_intersect, meta_data, true_genuses)
    
    if other is not None:
        (other_genuses, other_families) = other.classify_all(name_intersect, meta_data, true_genuses)
    
    print(number_of_equal_elements(orig_genuses, true_genuses) / len(orig_genuses))
    print(number_of_equal_elements(orig_families, true_families) / len(orig_families))
    
    if other is not None:
        print(number_of_equal_elements(true_genuses, other_genuses) / len(other_genuses))
        print(number_of_equal_elements(true_families, other_families) / len(other_families))

        print(number_of_equal_elements(orig_genuses, other_genuses) / len(orig_genuses))
        print(number_of_equal_elements(orig_families, other_families) / len(orig_families))


parser = argparse.ArgumentParser(description="")

parser.add_argument("--original_matrix", help="path to original distance matrix")
parser.add_argument("--original_dist",help="distance matrix")
parser.add_argument("--other_matrix", help="path to other distance matrix")
parser.add_argument("--other_dist",help="distance matrix of other")
parser.add_argument("--k", default=1, help="number of neighbors")

args = parser.parse_args()

with open(args.original_matrix, "rb") as f:
    (neighbors, names) = pickle.load(f)

with open(args.original_dist,"rb") as f:
    distances = pickle.load(f)

original = neighborMatrix(neighbors, names, distances, int(args.k))

if args.other_matrix is not None or args.other_dist is not None:
    with open(args.other_matrix, "rb") as f:
        (other_neighbors, other_names) = pickle.load(f)
    
    with open(args.other_dist,"rb") as f:
        other_distances = pickle.load(f)

    other = neighborMatrix(other_neighbors, other_names, other_distances, int(args.k))

    name_intersect = [name for name in names if name in other_names]
else:
    other = None
    name_intersect = names

meta_data = defaultdict(default_class,get_metadata_for(name_intersect, "../settings/db_config.json"))

classify_test(original, other, meta_data, name_intersect)
