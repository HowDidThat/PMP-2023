import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

dummy_data = np.loadtxt('./data/dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 2
x_1p = np.vstack([x_1**i for i
in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

with pm.Model() as model_l:
    alfa = pm.Normal('alfa', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10)
    eps = pm.HalfNormal('eps', 5)
    miu = alfa + beta * x_1s[0]
    y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)
with pm.Model() as model_p:
    alfa = pm.Normal('alfa', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    eps = pm.HalfNormal('eps', 5)
    miu = alfa + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)


x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
alfa_l_post = idata_l.posterior['alfa'].mean(("chain", "draw")).values
beta_l_post = idata_l.posterior['beta'].mean(("chain", "draw")).values
y_l_post = alfa_l_post + beta_l_post * x_new
plt.plot(x_new, y_l_post, 'C1', label='linear model')
alfa_p_post = idata_p.posterior['alfa'].mean(("chain", "draw")).values
beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = alfa_p_post + np.dot(beta_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()