
import matplotlib.pyplot as plt
import networkx as nx
import math
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random
import statistics

pointsSize = 10000


# volum de trafic , distributie poison
lam = 20 # clienti pe ora

# timp plasare si plata, distributie normala 

avgTimeCommand = 2 # minute
stdDerivation = 0.5 # minute

# Statie de gatit pregateste comanda in distributie exponentiala cu alfa minute
alfa = 2
traficVolume = stats.poisson.rvs(mu=20,size = pointsSize)

timePlaced = stats.norm.rvs(loc = 2, scale = 0.5, size = pointsSize)

alfaDistribution = stats.expon.rvs(loc = alfa,size = pointsSize)

timeAndAlfa = []

for i in range(pointsSize):
    timeAndAlfa.append (timePlaced[i] + alfaDistribution[i])
    
az.plot_posterior({"Trafic": traficVolume, "TimePlaced + cook": timeAndAlfa})
plt.show()

maxServingTime = 15
alfaStep = 0.5 # cat de aproape probam alfa

while True:
    alfa += alfaStep
    alfaDistribution = stats.expon.rvs(loc=alfa, size = pointsSize)
    timeAndAlfa = []
    for i in range(pointsSize):
        timeAndAlfa.append (timePlaced[i] + alfaDistribution[i])
    good = 0
    for i in timeAndAlfa:
        if i < maxServingTime:
            good += 1
    if good / len(timeAndAlfa) < 0.95:
        break
print ("Alfa to serve all clients in a Hour: " + str(alfa))


timePlaced = stats.norm.rvs(loc = 2, scale = 0.5, size = pointsSize)

alfaDistribution = stats.expon.rvs(loc = alfa,size = pointsSize)

cookTime = timePlaced + alfaDistribution

print("Avarege cooking time: " + str(statistics.mean(cookTime)))










