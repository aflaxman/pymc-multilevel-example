import pymc
import analysis1
reload(analysis1)
from pylab import hist, show
from pymc import Matplot

M = pymc.MCMC(analysis1)
M.use_step_method(pymc.AdaptiveMetropolis, [M.B, M.U, M.var_e1, M.var_u])
M.sample(iter=100000, burn=50000, thin=10, verbose=2)

fit = M.stats()
for k in fit.keys():
    print(k,fit[k]['mean'])

Matplot.plot(M)
