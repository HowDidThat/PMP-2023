import sys
import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor.tensor as pt


#Ex 1 

def posterior_grid(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

def posterior_grid1(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = []
    for i in grid:
        prior.append((i <= 0.5).astype(int))
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

def posterior_grid2(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = []
    for i in grid:
        prior.append(abs(i-0.5))
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior)
grid, posterior = posterior_grid1(points, h, t)
plt.plot(grid, posterior)
grid, posterior = posterior_grid2(points, h, t)
plt.plot(grid, posterior)


plt.show()

#Ex 2

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    return error

N_values = [100, 500, 1000, 5000,10000]

mean_errors = []
std_dev_errors = []

for N in N_values:
    errors = [estimate_pi(N) for _ in range(100)]  

    mean_error = np.mean(errors)
    std_dev_error = np.std(errors)

    mean_errors.append(mean_error)
    std_dev_errors.append(std_dev_error)

plt.errorbar(N_values, mean_errors, yerr=std_dev_errors, fmt='o-', capsize=5)
plt.xlabel('Numărul de puncte (N)')
plt.ylabel('Eroare (%)')
plt.show()


#Ex 3

def metropolis(func, draws=10000):
    """A very simple Metropolis implementation"""
    trace = np.zeros(draws)
    old_x = 0.5 # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


