import pymc as pm
import arviz as az
import numpy as np

import matplotlib.pyplot as plt


theta= [0.2, 0.5]
yValues = [0, 5, 10]
lambdaPoison = 10  

traces = []
for i in range(2):
    for j in range(3):
        m = pm.Model()
        with m:
            
            n0 = pm.Poisson('n {}{}'.format(i,j), mu=lambdaPoison)
            
            likelihood00 = pm.Binomial('prob00', n=n0, p=theta[i], observed=yValues[j])
            
            trace1 = pm.sample(5000, tune=2000, cores=1)
        traces.append(trace1)


az.plot_posterior(traces[0], var_names='n 00', figsize=(4, 3), hdi_prob=0.95, round_to=2)
az.plot_posterior(traces[1], var_names='n 01', figsize=(4, 3), hdi_prob=0.95, round_to=2)
az.plot_posterior(traces[2], var_names='n 02', figsize=(4, 3), hdi_prob=0.95, round_to=2)
az.plot_posterior(traces[3], var_names='n 10', figsize=(4, 3), hdi_prob=0.95, round_to=2)
az.plot_posterior(traces[4], var_names='n 11', figsize=(4, 3), hdi_prob=0.95, round_to=2)
az.plot_posterior(traces[5], var_names='n 12', figsize=(4, 3), hdi_prob=0.95, round_to=2)
plt.show()
