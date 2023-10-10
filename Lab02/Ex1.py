import math
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random
import statistics


x = 10

mecanic1 = stats.expon.rvs(scale = 1/4,size = 10000)
mecanic2 = stats.expon.rvs(scale = 1/6,size = 10000)

mecanic1 += x
mecanic2 += x
mecanic3 = []
for element in range(len(mecanic1)):
    r = random.random()
    if (r < 0.4):
        mecanic3.append(mecanic1[element])
    else:
        mecanic3.append(mecanic2[element])

print("The mean is : ", np.mean(mecanic3))
print("Standard derivation: ", statistics.stdev(mecanic3))

az.plot_dist(mecanic3)

plt.show()
