import argparse
import pickle
import numpy as np

from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)
from util.numberOfEqualElements import number_of_equal_elements


def x(orig_matrix, orig_names, other_matrix, other_names, args):

    orig_neighbors = orig_matrix[:, 1]
    other_neighbors = other_matrix[:, 1]
    found_organisms = {
        name: orig_names[neighbor] for name, neighbor in zip(orig_names, orig_neighbors)
    }
    other_organisms = {
        name: other_names[neighbor]
        for name, neighbor in zip(other_names, other_neighbors)
    }

    name_intersect = [name for name in orig_names if name in other_names]

    meta_data = get_metadata_for(orig_names, "db_config.json")

    true_genuses = [meta_data[point]["genus"] for point in name_intersect]
    true_families = [meta_data[point]["family"] for point in name_intersect]
    orig_genuses = []
    other_genuses = []
    orig_families = []
    other_families = []
    print("metadata ready")
    for point in name_intersect:
        try:
            orig_genuses.append(meta_data[found_organisms[point]]["genus"])
            orig_families.append(meta_data[found_organisms[point]]["family"])
            other_genuses.append(meta_data[other_organisms[point]]["genus"])
            other_families.append(meta_data[other_organisms[point]]["family"])
        except Exception:
            print("fail")
            other_genuses.append("none")
            other_families.append("none")
            continue

    print(number_of_equal_elements(orig_genuses, true_genuses) / len(orig_genuses))
    print(number_of_equal_elements(orig_families, true_families) / len(orig_families))

    print(number_of_equal_elements(true_genuses, other_genuses) / len(other_genuses))
    print(number_of_equal_elements(true_families, other_families) / len(other_families))

    print(number_of_equal_elements(orig_genuses, other_genuses) / len(orig_genuses))
    print(number_of_equal_elements(orig_families, other_families) / len(orig_families))


# ha två matriser
# för varje element
#   vilken är NN/3 NN
#   i vilken utsträckning är det en hit/miss


parser = argparse.ArgumentParser(description="")

parser.add_argument("--original_matrix", help="path to original distance matrix")
parser.add_argument("--other_matrix", help="path to other distance matrix")
parser.add_argument("--k", default=1, help="number of neighbors")

args = parser.parse_args()

# give both name lists, otherwise you cannot translate.

with open(args.original_matrix, "rb") as f:
    (neighbors, names) = pickle.load(f)

with open(args.other_matrix, "rb") as f:
    (other_neighbors, other_names) = pickle.load(f)

x(neighbors, names, other_neighbors, other_names, args)
