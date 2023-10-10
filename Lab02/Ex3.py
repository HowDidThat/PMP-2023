import math
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random
import statistics

def fairCoin():
    r = random.random()
    if r < 0.5:
        return "s"
    return "b"

def unfairCoin():
    r = random.random()
    if r < 0.3:
        return "s"
    return "b"

results = []

for i in range(100):
    for i in range (10):
        results.append(fairCoin()+unfairCoin())

outcomes = ["ss","sb","bs","bb"]
numberOutcomes = []
for i in outcomes:
    numberOutcomes.append(results.count(i))

for i in range(len(outcomes)):
    print(outcomes[i]," : ",numberOutcomes[i])

plt.bar(height=numberOutcomes,x = outcomes)
plt.show()