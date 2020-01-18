"""

* same nearest neighbor is rare (in rosen, asymgauss, multishell for <=8 steps)

* likelihood rank difference between chain start and end is 
  typically ~ 117 (expectation from 400 live points)
  in a converged chain (>= 8 steps in rosen, >= 64 steps in asymgauss, >= 8 in multishell)

* distance from start to end > maxradius 
  in a converged chain (>= 64 steps in rosen, >= 64 steps in asymgauss, >= 64 in multishell)

* angle from start to end > 80° (dot product is near zero)
  specifically, 1 sigma lower uncertainty is 90-55 / (ndims-1)**0.5 degrees
  in a converged chain (>= 64 steps in rosen, > 64 steps in asymgauss, >= 64 in multishell)
  

Possible strategies:

* if above(below) the critical value, increase(decrease) nsteps
  --> when median is reached by nsteps, there is an equilibrium of increases and decreases

* evaluate convergence at 50% of chain completeness



"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal

Nmax = None

figdL = plt.figure("dL").gca()
figdist = plt.figure("dist").gca()
figangle = plt.figure("angle").gca()
figindex = plt.figure("index").gca()
fignsteps = plt.figure("nsteps").gca()

for filename in sys.argv[1:]:
	print("loading '%s'... " % filename)
	data = np.loadtxt(filename)[:]
	ndim = (data.shape[1] - 1 - 3 - 4) // 4
	Lmin = data[:,0]
	nsteps, maxradius, mean_pair_distance, iLstart, iLfinal, itstart, itfinal = data[:,1+ndim*4:].transpose()
	ustart = data[:,1+ndim*0:1+ndim*1]
	ufinal = data[:,1+ndim*1:1+ndim*2]
	tstart = data[:,1+ndim*2:1+ndim*3]
	tfinal = data[:,1+ndim*3:1+ndim*4]
	
	label = filename.split('/')[-2]
	#figdist.plot(((tfinal - tstart)**2).sum(axis=1)**0.5, label='dist %s' % label)
	l, = figdist.plot(scipy.signal.medfilt(((tfinal - tstart)**2).sum(axis=1)**0.5, 401), label='dist %s' % label)
	figdist.plot(maxradius, ls='--', label='radius %s' % label, color=l.get_color())
	#figdL.plot(mean_pair_distance, label='pair %s' % label)
	
	angle = np.arccos( (tfinal * tstart).sum(axis=1) / (((tstart**2).sum(axis=1) * (tfinal**2).sum(axis=1)))**0.5) / np.pi * 180
	l, = figangle.plot(angle, label='%s' % label, alpha=0.2)
	figangle.plot(scipy.signal.medfilt(angle, 41), ls='--', color=l.get_color())
	
	#l, = figdL.plot(Lmin[1:]-Lmin[:-1], label='dL %s' % label, lw=1, alpha=0.25)
	#figdL.plot(scipy.signal.medfilt(Lmin[1:]-Lmin[:-1], 401), color=l.get_color())
	figdL.plot(scipy.signal.medfilt(Lmin[1:]-Lmin[:-1], 401)*400, label='dL %s' % label)
	figdL.plot(Lmin[1:]-Lmin[0], ls='--', label='L %s' % label)
	
	#figindex.plot(iLstart, ls='-', label='Lstart %s' % label, alpha=0.2)
	#figindex.plot(iLfinal, ls='-', label='Lfinal %s' % label, alpha=0.2)
	#figindex.plot(np.abs(iLfinal-iLstart), ls='-', label='Ldiff %s' % label, alpha=0.2)
	l, = figindex.plot(scipy.signal.medfilt(np.abs(iLfinal-iLstart), 401), ls='--', label='Ldiff %s' % label)
	figindex.plot((itstart == itfinal)*400, ls='-', label='same NN %s' % label, alpha=0.2, color=l.get_color())
	#figindex.plot(scipy.signal.medfilt((itstart == itfinal)*400, 401), ls='--', label='same NN %s' % label)
	#figindex.plot(scipy.ndimage.filters.gaussian_filter((itstart == itfinal)*400, 400)*400, ls='--', label='same NN %s' % label)
	
	fignsteps.plot(nsteps, ls='-', label='%s' % label)

a = np.random.randint(0, 400, size=100000)
b = np.random.randint(0, 400, size=100000)
print('delta_index = %.1f, same=%.2f%%' % (np.median(np.abs(a - b)), (a==b).mean()*100), 1/400.)

figindex.set_xlim(0, Nmax)
figindex.set_yscale('log')
figindex.hlines(117, *figindex.get_xlim())
figindex = plt.figure("index")
plt.legend(loc='best')
figindex.savefig("evolution_index.pdf", bbox_inches='tight')
plt.close()


figdist.set_xlim(0, Nmax)
#figdist.set_yscale('log')
figdist.set_ylabel('Distance')
figdist = plt.figure("dist")
plt.legend(loc='best')
figdist.savefig("evolution_dist.pdf", bbox_inches='tight')
plt.close()



fignsteps.set_xlim(0, Nmax)
#fignsteps.set_yscale('log')
fignsteps.set_ylabel('Number of steps')
fignsteps = plt.figure("nsteps")
plt.legend(loc='best')
fignsteps.savefig("evolution_nsteps.pdf", bbox_inches='tight')
plt.close()

figangle.set_xlim(0, Nmax)
figangle.set_ylim(0, 180)
figangle.set_ylabel('Angle [deg]')
figangle.hlines(90-55 / (ndim-1)**0.5, *figangle.get_xlim())
figangle = plt.figure("angle")
plt.legend(loc='best')
figangle.savefig("evolution_angle.pdf", bbox_inches='tight')
plt.close()

figdL.set_xlim(0, Nmax)
figdL.set_ylim(0.1, None)
figdL.set_ylabel('Likelihood difference')
#figdL.set_yscale('log')
figdL = plt.figure("dL")
plt.legend(loc='best')
figdL.savefig("evolution_dL.pdf", bbox_inches='tight')
plt.close()


