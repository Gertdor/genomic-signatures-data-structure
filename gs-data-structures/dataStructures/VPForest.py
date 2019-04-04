from dataStructures.VPTree import VPTree
from operator import itemgetter
import numpy as np


class VPForest:
    def __init__(self, values, random, max_leaf_size=1):
        print(random)
        self.forest = VPForest._create_VP_forest(values, random, max_leaf_size)

    def _create_VP_forest(values, random, max_leaf_size):

        paired_values = [(value.get_gc(), value) for value in values]
        # sorted_values.sort(key=itemgetter(0))
        cutoff_points = [0] + np.arange(0.25, 0.76, 0.02).tolist() + [1]
        cutoff_ranges = zip(cutoff_points[0:-1], cutoff_points[1:])
        span = 0.12

        forest = {
            (low, high): VPForest._generate_tree(
                paired_values, low, high, span, random, max_leaf_size
            )
            for low, high in cutoff_ranges
        }
        return forest

    def _generate_tree(values, low_gc, high_gc, span, random, max_leaf_size):

        values_to_use = [
            value[1]
            for value in values
            if value[0] > low_gc - span and value[0] < high_gc + span
        ]
        return VPTree(values_to_use, random, max_leaf_size)

    def nearest_neighbor(self, point, k=1, greedy_factor=1, gc_pruning=False):
        # TODO Can do some smarter implementation of the keys here, if it is slow

        gc_content = point.get_gc()
        for key in self.forest:
            if gc_content > key[0] and gc_content < key[1]:
                return self.forest[key].nearest_neighbor(
                    point, k, greedy_factor, gc_pruning
                )
