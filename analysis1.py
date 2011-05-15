import pylab as pl
import numpy as np
import pymc

run1 = pl.csv2rec('run1.csv')
pn_data = np.genfromtxt('run1.csv', delimiter=",", unpack=True, skip_header=1)

pid = pn_data[0] 	#subject id
assert pl.all(run1.i == pid)

gid = pn_data[1].astype(int) 	#group (cluster) id
assert pl.all(run1.j == gid)

tx = pn_data[2].reshape(200,1) 	#treatment condition (1 = clustered, 0 = unclustered)
assert pl.all(run1.treat == tx.T)

y = pn_data[3] 	#outcome variable
assert pl.all(run1.y == y)

##Design matrix for intercept and intervention effect
inter = np.ones((len(pid),1), dtype=int)
X = np.hstack((inter,tx))

X2 = [[1., t_ij] for t_ij in run1.treat]
assert pl.all(X == X2)

##Design matrix for cluster effect
Z = (gid[:,None] == np.unique(gid)).astype(int)
##Getting rid of the columns for the unclustered "clusters of 1"
Z = Z[0:240:1,0:10:1]

Z2 = pl.zeros([len(run1), 10])
for row, j in enumerate(run1.j):
    if j <= 10:
        Z2[row, j-1] = 1.
assert pl.all(Z == Z2)

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
def y_i(value=y, mu=y_hat, tau=tau_e1):
    return pymc.normal_like(value,mu,tau)


