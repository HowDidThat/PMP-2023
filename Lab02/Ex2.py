import math
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random
import statistics

nrClients = 10000

server1 = stats.gamma.rvs(4,scale = 1/3,size=nrClients)
server2 = stats.gamma.rvs(4,0,1/2,size=nrClients)
server3 = stats.gamma.rvs(5,0,1/2,size=nrClients)
server4 = stats.gamma.rvs(5,0,1/3,size=nrClients)
latenta = stats.expon.rvs(0,1/4,size=nrClients)

server1 += latenta
server2 += latenta
server3 += latenta
server4 += latenta

latencies = []
for i in range(len(server1)):
    r = random.random()
    if (r < 0.25):
        latencies.append(server1[i])
    elif (r < 0.5):
        latencies.append(server2[i])
    elif (r < 0.8):
        latencies.append(server3[i])
    else:
        latencies.append(server4[i])

s = 0
for i in latencies:
    if i > 3:
        s+=1

print("Probability: ", s/len(latencies))

plt.hist(latencies)
plt.show()