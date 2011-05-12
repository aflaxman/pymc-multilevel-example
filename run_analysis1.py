import pymc
import analysis1
from pylab import hist, show
from pymc import Matplot

M = pymc.MCMC(analysis1)
M.sample(iter=100000, burn=10000, thin=10, verbose=2)

fit = M.stats()
for k in fit.keys():
    print(k,fit[k]['mean'])

Matplot.plot(M)
