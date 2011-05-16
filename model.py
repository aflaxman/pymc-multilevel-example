import pylab as pl
import pymc as mc

data = pl.csv2rec('data.csv')

## Design matrix for intercept and intervention effect
X = [[1., t_ij] for t_ij in data.treat]

## Design matrix for cluster effect
Z = pl.zeros([len(data), 10])
for row, j in enumerate(data.j):
    if j <= 10:
        Z[row, j-1] = 1.

## Priors
var_u = mc.Gamma('var_u', alpha=1, beta=1, value=1.)
tau_u = mc.Lambda('tau_u', lambda v=var_u: v**-1, trace=False)

B = mc.Normal('B', mu=[0, 0], tau=10000**-1)

U = mc.Normal('u', mu=0, tau=tau_u, value=pl.zeros(10))

var_e1 = mc.Uniform('var_e1', lower=0, upper=100, value=[1., 1.])
tau_e1 = mc.Lambda('tau_e1', lambda v=var_e1: v**-1, trace=False)

@mc.deterministic(trace=False)
def y_hat(B=B, X=X, Z=Z, U=U):
    return pl.dot(X,B) + pl.dot(Z,U)

@mc.stochastic(observed=True)
def y_i(value=data.y, mu=y_hat, tau=tau_e1):
    return mc.normal_like(value,mu,tau[data.treat])
