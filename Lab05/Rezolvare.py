# distributie poisson cu alfa necunoscut

# modificari trafic : ora 7,16 crestere,   ora 8,19 scadere

import numpy as np
import pymc as pm
import csv
import matplotlib.pyplot as plt
data = []
with open('trafic.csv',newline='') as csvfile:
    csvData = csv.reader(csvfile,delimiter = ',')
    head = True
    for row in csvData:
        if row[1].isnumeric():
            data.append(int(row[1]))

basic_model = pm.Model()

times = [0,180,240,720,900,1200]

with basic_model:
    pTotal = []
    p1 = pm.Poisson("4-7",mu=0)
    p2 = pm.Poisson("7-8",mu=0)
    p3 = pm.Poisson("8-14",mu=0)
    p4 = pm.Poisson("14-19",mu=0)
    p5 = pm.Poisson("19-24",mu=0)

    mu1 = np.mean(data[times[0]:times[1]])
    mu2 = np.mean(data[times[1]:times[2]])
    mu3 = np.mean(data[times[2]:times[3]])
    mu4 = np.mean(data[times[3]:times[4]])
    mu5 = np.mean(data[times[4]:times[5]])
    ObsP1 = pm.Poisson("ObsP1",mu = mu1, observed = data[times[0]:times[1]])
    ObsP2 = pm.Poisson("ObsP2",mu = mu2, observed = data[times[1]:times[2]])
    ObsP3 = pm.Poisson("ObsP3",mu = mu3, observed = data[times[2]:times[3]])
    ObsP4 = pm.Poisson("ObsP4",mu = mu4, observed = data[times[3]:times[4]])
    ObsP5 = pm.Poisson("ObsP5",mu = mu5, observed = data[times[4]:times[5]])
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step, return_inferencedata=False, chains=1,cores=1)

print(trace["7-8"])

    