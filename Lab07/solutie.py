import pymc as pm
import arviz as az
import numpy as np
import csv
import matplotlib.pyplot as plt

import pandas as pd

allData = [[],[],]

with open("Lab07//auto-mpg.csv",newline="") as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        if (row['mpg'] != "?" and row['horsepower'] != '?'):
            allData[0].append(float(row['mpg']))
            allData[1].append(float(row['horsepower']))



data = {"MPG":allData[0],"HP":allData[1]}

dataFrame = pd.DataFrame(data)
dataFrame.plot(x='MPG',y='HP',kind='scatter')
plt.xlabel('MPG')
plt.ylabel('HP')
#plt.show()


with pm.Model():
    alfa = pm.Normal('alfa',mu=0,sigma = 5)
    beta = pm.Normal('beta',mu=0,sigma = 3)
    error = pm.Normal('error', mu=0,sigma = 0.1)
    mpg = pm.Deterministic('mpg', alfa + beta * data['HP'] + error)
    yPredictie = pm.Normal('yPredictie',mu=mpg,sigma=error,observed=data['MPG'])

    trace = pm.sample(5000,tune=2000,cores=1,return_inferencedata=True)

'''
post = trace.posterior.stack(samples={"alfa","beta","error"})
betaM = post['beta'].mean().item()
alfaM = post['alfa'].mean().item()
errorM = post['error'].mean().item()
'''

alfaM = trace.posterior['alfa'].mean().item()
betaM = trace.posterior['beta'].mean().item()
errorM = trace.posterior['error'].mean().item()
print ("Alfa: ",alfaM)
print ("Beta: ",betaM)
print ("Error: ",errorM)
print("Line: ", f'y = {alfaM:.2f} + {betaM:.2f} * x+ {errorM:.2f}')

plt.show()


