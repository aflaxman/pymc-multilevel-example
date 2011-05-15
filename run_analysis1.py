import pymc as mc
import analysis1
reload(analysis1)
import pylab as pl
#from pymc import Matplot
import Matplot
reload(Matplot)

# Generate reasonable initial values by maximizing posterior with
# variances held fixed
M = mc.MAP([analysis1.B, analysis1.U, analysis1.y_i])
M.fit(method='fmin_powell', verbose=0)

# Sample from full posterior distribution
M = mc.MCMC(analysis1)
M.use_step_method(mc.AdaptiveMetropolis, [M.B, M.U, M.var_e1, M.var_u])
M.sample(iter=40000, burn=20000, thin=20, verbose=0)

fit = M.stats()
for k in sorted(fit.keys()):
    print '%10s: %s' % (k, pl.floor(fit[k]['mean']*100.)/100.)

Matplot.plot(M)
pl.show()
