import numpy as np
from scipy import stats


def distance_function_stats(elements):
    distances = np.array(
        [current.distance(other) for current in elements for other in elements]
    )
    print("Stat summary:", stats.describe(distances), "\n")
