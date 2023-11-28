import pymc as pm
import arviz as az
import numpy as np
import csv
import matplotlib.pyplot as plt
import csv
import math

numarVanzari = 500

data = []

with open('Lab09\\Prices.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    next(csvFile)
    for lines in csvFile:
        data.append(lines)
freq = [int(x[1]) for x in data]
hard = [math.log(int(x[2])) for x in data]
price = [int(x[0]) for x in data]

with pm.Model() as regresionModel:
    alfa = pm.Normal('alfa',mu=0,sigma=10)
    beta1 = pm.Normal('beta1',mu=0,sigma=1)
    beta2 = pm.Normal('beta2',mu=0,sigma=2)
    eps = pm.HalfCauchy('eps', 5)
    miu = pm.Deterministic('niu', alfa + freq * beta1 + hard * beta2)
    pricePredict = pm.Normal("pricePredict",mu=miu,sigma=eps,observed=price)
    idata = pm.sample(2000,tune=100, return_inferencedata= True, cores=1)


posteriorData = idata['posterior']
alfaM = posteriorData['alfa'].mean().item()
beta1M = posteriorData['beta1'].mean().item()
beta2M = posteriorData['beta2'].mean().item()

print(f"alfaM: {alfaM}\nbeta1M: {beta1M}\nbeta2M: {beta2M}")

az.plot_posterior(posteriorData, var_names=['beta1', 'beta2'], hdi_prob=0.95)
plt.show()

#3 cum beta1 si beta2 sunt diferite de 0 atunci frecventa procesorului si memori influenteaza pretul final al procesorului
