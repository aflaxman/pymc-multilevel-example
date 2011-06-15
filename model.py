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
B = mc.Normal('B', mu=[0, 0], tau=10000**-1)
U = mc.Normal('u', mu=0, tau=var_u**-1, value=pl.zeros(10))
var_e1 = mc.Uniform('var_e1', lower=0, upper=100, value=[1., 1.])

## Systematic Model
y_hat = mc.LinearCombination('X*B', [X], [B]) + mc.LinearCombination('Z*U', [Z], [U])

## Stochastic Model
y_i = mc.Normal('y_i', value=data.y, mu=y_hat, tau=var_e1[data.treat]**-1, observed=True)
