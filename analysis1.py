import numpy as np
import pymc

pn_data = np.genfromtxt('run1.csv', delimiter=",", unpack=True, skip_header=1)

pid = pn_data[0] 	#subject id
gid = pn_data[1].astype(int) 	#group (cluster) id
tx = pn_data[2].reshape(200,1) 	#treatment condition (1 = clustered, 0 = unclustered)
y = pn_data[3] 	#outcome variable

##Design matrix for intercept and intervention effect
inter = np.ones((len(pid),1), dtype=int)
X = np.hstack((inter,tx))

##Design matrix for cluster effect
Z = (gid[:,None] == np.unique(gid)).astype(int)
##Getting rid of the columns for the unclustered "clusters of 1"
Z = Z[0:240:1,0:10:1]

## Priors
var_u = pymc.Gamma('var_u', alpha=1, beta=1)
tau_u = pymc.Lambda('tau_u', lambda v=var_u: v**-1)

b0 = pymc.Normal('b0', mu=0, tau=10000**-1)
b1 = pymc.Normal('b1', mu=0, tau=10000**-1)

U = pymc.Normal('u', mu=0, tau=tau_u, value=np.zeros(10))

var_e1 = pymc.Uniform('var_e1', lower=0, upper=100)
tau_e1 = pymc.Lambda('tau_e1', lambda v=var_e1: v**-1)

@pymc.deterministic
def y_hat(b0=b0, b1=b1, X=X, Z=Z, U=U):
	B = np.array((b0,b1))
	return np.dot(X,B)+np.dot(Z,U)

@pymc.stochastic(observed=True)
def y_i(value=y, mu=y_hat, tau=tau_e1):
	return pymc.normal_like(value,mu,tau)


