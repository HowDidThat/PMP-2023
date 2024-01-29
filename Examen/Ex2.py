import pymc as pm
import arviz as az
import numpy as np
import csv
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd
from scipy import stats



def posterior_grid(grid_points=50, n=5):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.geomspace(1, 100, grid_points)
    prior = np.repeat(100/grid_points, grid_points)
    likelihood = stats.geom.pmf(n,grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

grid, posterior = posterior_grid(50, 5)
plt.plot(grid, posterior, 'o-')
plt.yticks([])
plt.xlabel('θ')
plt.show()

print(grid)
print(posterior)