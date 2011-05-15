import pymc
import analysis1
reload(analysis1)
import pylab as pl
#from pymc import Matplot
import Matplot
reload(Matplot)

# Generate reasonable initial values by maximizing posterior with
# variances held fixed
M = pymc.MAP([analysis1.B, analysis1.U, analysis1.y_i])
M.fit(method='fmin_powell', verbose=1)

# Sample from full posterior distribution
M = pymc.MCMC(analysis1)
M.use_step_method(pymc.AdaptiveMetropolis, [M.B, M.U, M.var_e1, M.var_u])
M.sample(iter=20000, burn=10000, thin=10, verbose=2)

fit = M.stats()
for k in fit.keys():
    print '%10s: %s' % (k, pl.floor(fit[k]['mean']*100.)/100.)

Matplot.plot(M)
pl.show()
