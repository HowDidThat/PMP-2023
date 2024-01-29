import pymc as pm
import arviz as az
import numpy as np
import csv
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd
from scipy.stats import norm
#importing ther dataFrame from the csv
data = pd.read_csv('BostonHousing.csv')

#separating the columns that need to be kept
keptColumns = ["medv","rm","crim","indus"]
#removing the other columns from the dataFrame
data = data[keptColumns]
print(data)

print(data)



with pm.Model() as model:
    #Creating the model using only the required data: rm,crim,indus as independent variables and medv as a dependent variable
    alfa = pm.Normal('alpha',mu=0,sigma=10)
    rm = pm.Normal("rm",mu=0,sigma=100)
    crim = pm.Normal("crim",mu=0,sigma=10)
    indus = pm.Normal("indus",mu=0,sigma=100)   
    sigma = pm.HalfCauchy("sigma",beta=20)

    medv = pm.Normal("medv",mu = alfa + rm*data["rm"] + crim * data["crim"] + indus * data["indus"], sigma=sigma, observed=data['medv'])
    idata_prem = pm.sample(2000, return_inferencedata=True,cores=1)


# creatin a 3 plot window for the 3 hdi
fig, axs = plt.subplots(3)

#plotting the 3 
az.plot_forest(idata_prem, hdi_prob=0.95, var_names=["rm"], ax=axs[0])
az.plot_forest(idata_prem, hdi_prob=0.95, var_names=["crim"], ax=axs[1])
az.plot_forest(idata_prem, hdi_prob=0.95, var_names=["indus"], ax=axs[2])
plt.tight_layout()
plt.savefig("Ex1")

ppc = pm.sample_posterior_predictive(model, samples=idata_prem, var_names=["medv"])

hdi_50 = az.hdi(ppc["medv"].flatten(), hdi_prob=0.5)
print(f"50% HDI for medv: {hdi_50}")