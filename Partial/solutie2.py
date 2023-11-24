import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

#punctele se vor intalni in jurul valorii 20 , timp mediu de asteptare
mu = 20
#majoritatea vor fi inte 16-24
sigma = 4

# o distributie normala a timpului de asteptare
timpMediu = np.random.normal(loc = mu,scale = sigma,size=200)

with pm.Model() as model:
    # deoarece timpul de asteptare este o distributie normala, putem alege o distributie
    # normala si una semi normala
    sigma = pm.HalfNormal('sigma', sigma=4)
    miu = pm.Normal('miu', mu=20) 
    nor = pm.Normal('nor', mu=miu, sigma=sigma, observed=timpMediu) 
    #facem inferenta pe modelul de mai sus 
    trace = pm.sample(1000, tune=2000, cores = 1,return_inferencedata=True)
# afisam din inferenta graful posteriori
posterior_data = trace["posterior"]
ppc = pm.sample_posterior_predictive(trace, model=model)
plt.plot(ppc)
plt.show()