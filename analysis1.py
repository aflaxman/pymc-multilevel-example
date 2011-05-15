import pylab as pl
import numpy as np
import pymc

run1 = pl.csv2rec('run1.csv')

##Design matrix for intercept and intervention effect
X = [[1., t_ij] for t_ij in run1.treat]

##Design matrix for cluster effect
Z = pl.zeros([len(run1), 10])
for row, j in enumerate(run1.j):
    if j <= 10:
        Z[row, j-1] = 1.

## Priors
var_u = pymc.Gamma('var_u', alpha=1, beta=1, value=1.)
tau_u = pymc.Lambda('tau_u', lambda v=var_u: v**-1)

B = pymc.Normal('B', mu=[0, 0], tau=10000**-1)

U = pymc.Normal('u', mu=0, tau=tau_u, value=np.zeros(10))

var_e1 = pymc.Uniform('var_e1', lower=0, upper=100, value=1.)
tau_e1 = pymc.Lambda('tau_e1', lambda v=var_e1: v**-1)

@pymc.deterministic(trace=False)
def y_hat(B=B, X=X, Z=Z, U=U):
    return np.dot(X,B)+np.dot(Z,U)

@pymc.stochastic(observed=True)
def y_i(value=run1.y, mu=y_hat, tau=tau_e1):
    return pymc.normal_like(value,mu,tau)


