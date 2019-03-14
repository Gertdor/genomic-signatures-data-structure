from scipy import stats
from random import randint
from dataStructures.VPTreeElement import VPTreeElement
import numpy as np

numberList = [VPTreeElement(np.random.uniform(0, 0.35, 2)) for x in range(5000)]
distances = [current.distance(other) for current in numberList for other in numberList]
print(stats.describe(distances))
