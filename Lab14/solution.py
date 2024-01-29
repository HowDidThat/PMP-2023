import pymc as pm
import arviz as az
import numpy as np
import csv
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd

#Ex 1
modelCentrat = az.load_arviz_data("centered_eight")
modelNecentrat = az.load_arviz_data("non_centered_eight")

az.plot_posterior(modelCentrat)
plt.savefig("1_centrat.png")
plt.close()
az.plot_posterior(modelCentrat)
plt.savefig("1_necentrat.png")
plt.close()

print("Model centrat: ")
print(f"    Lanturi:  {modelCentrat.posterior.chain.shape[0]}")
print(f"    Marime totala: {modelCentrat.posterior.chain.shape[0] * modelCentrat.posterior.draw.shape[0]}")


print("Model necentrat: ")
print(f"    Lanturi:  {modelNecentrat.posterior.chain.shape[0]}")
print(f"    Marime totala: {modelNecentrat.posterior.chain.shape[0] * modelCentrat.posterior.draw.shape[0]}")

#Ex 2
print("Comparatie MU")

summaries = pd.concat([az.summary(modelCentrat, var_names=['mu']),az.summary(modelNecentrat, var_names=['mu'])])
summaries.index = ['centrat', 'necentrat']
print(summaries)

print("Comparatie TAU")

summaries = pd.concat([az.summary(modelCentrat, var_names=['tau']),az.summary(modelNecentrat, var_names=['tau'])])
summaries.index = ['centrat', 'necentrat']
print(summaries)



az.plot_autocorr(modelCentrat, var_names=["mu", "tau"], combined=True)
plt.savefig("2_centrat")
plt.close()
az.plot_autocorr(modelNecentrat, var_names=["mu", "tau"], combined=True)
plt.savefig("2_necentrat")
plt.close()

divergenteCentrat = modelCentrat.sample_stats.diverging.sum()
divergenteNecentrat = modelNecentrat.sample_stats.diverging.sum()

print("Divergente model centrat: ", divergenteCentrat)
print("Divergente model necentrat: ", divergenteNecentrat)

az.plot_pair(modelCentrat, var_names=["mu", "tau"], divergences=True)
plt.savefig("3_centrat")
plt.close()

az.plot_pair(modelNecentrat, var_names=["mu", "tau"], divergences=True)
plt.savefig("3_necentrat")
plt.close()

