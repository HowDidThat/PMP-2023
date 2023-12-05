import pymc as pm
import arviz as az
import numpy as np
import csv
import matplotlib.pyplot as plt
import csv
import math


data = []
with open('Lab10\\Admission.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    next(csvFile)
    for lines in csvFile:
        data.append(lines)
gre = [int(x[1]) for x in data]
gpa = [float(x[2]) for x in data]
admitted = [int(x[0]) for x in data]


def normalizeData(allData):
    minValue = min(allData)
    maxValue = max(allData)
    return [ (x-minValue)/(maxValue - minValue) for x in allData ]

gre = normalizeData(gre)
gpa = normalizeData(gpa)

with pm.Model() as newModel:
    beta0 = pm.Normal("beta0", mu=0,sigma=10)
    beta1 = pm.Normal("beta1", mu=1,sigma=10)
    beta2 = pm.Normal("beta2", mu=2,sigma=10)
    
    miu = beta0 + gre*beta1 + gpa*beta2
    theta = pm.Deterministic('theta',1/(1+pm.math.exp(-miu)))
    bd = pm.Deterministic("bd",-beta0/beta1-beta1/beta2 * gre)

    admissionPredict = pm.Bernoulli("admissionPredict",p=theta, observed=admitted)
    idata = pm.sample(1000,return_inferencedata=True,cores = 1)

idx = np.argsort(gre)
bd = idata.posterior['bd'].mean(("chain", "draw"))[idx]
plt.scatter(gre,gpa)
plt.plot(gre,bd,color='k')
az.plot_hdi(gpa,idata.posterior['bd'],color='k')
plt.xlabel("gpa")
plt.ylabel("gre")
plt.show()


