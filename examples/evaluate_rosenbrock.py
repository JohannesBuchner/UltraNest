import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, log
import sys
import json
import scipy.stats, scipy.special


def shell_vol(ndim, r, w):
    # integral along the radius
    mom = scipy.stats.norm.moment(ndim - 1, loc=r, scale=w)
    # integral along the angles is surface of hyper-ball
    # which is volume of one higher dimension x (ndim + 1)
    vol = pi**((ndim)/2.) / scipy.special.gamma((ndim)/2. + 1)
    surf = vol * ndim
    return mom * surf


methods = {}
for filename in sys.argv[1:]:
    info = json.load(open(filename))
    ncall = info['ncall']
    logz = info['logz']
    logzerr = info['logzerr']
    ndim = len(info['paramnames'])
    path = filename.replace('/info/results.json', '')
    adaptation = ''
    if '-adapt' in path:
        path, adaptation = path.split('-adapt')
        adaptation = '-' + adaptation
    
    parts = path.split('-')
    if len(parts) == 3:
        method = parts[-1].strip('1234567890') + adaptation
    else:
        method = 'MLFriends'
    
    method_data = methods.get(method, [])
    method_data.append([ncall, logz, logzerr])
    methods[method] = method_data
    print(ncall, logz, path)

for method, data in sorted(methods.items()):
    x, y, yerr = np.transpose(data)
    i = np.argsort(x)
    x, y, yerr = x[i], y[i], yerr[i]
    print(method, x, y, yerr)
    plt.errorbar(x=x, y=y, yerr=yerr, marker='x', label=method)
if 'rosen' in filename and ndim == 50:
    plt.plot(1.3e9, -288.6, 'o', label='DNest4')

if 'rosen' in filename:
    true_logz_lo = 3.7 + ndim*-5.7 + ndim**2 * -0.01
    true_logz_hi = 3.1 + ndim*-5.7 + ndim**2 * 0.01
    true_logz_lo = 3.85689+0.3732 + (-5.82502 - 0.139) * ndim + (0.00417525-0.01149) * ndim**2
    true_logz_hi = 3.85689-0.3732 + (-5.82502 + 0.139) * ndim + (0.00417525+0.01149) * ndim**2
    #true_logz_lo = 3.7 + ndim*-5.7 + ndim**2 * -0.01
    #true_logz_hi = 3.7 + ndim*-5.7
    problemname = 'rosenbrock'
elif 'multishell' in filename:
    r = 0.2
    w = 0.001 / ndim
    r1, r2 = r, r
    w1, w2 = w, w
    Z_analytic = log(shell_vol(ndim, r1, w1) + shell_vol(ndim, r2, w2))
    true_logz_lo, true_logz_hi = Z_analytic, Z_analytic
    problemname = 'multishell'
elif 'asymgauss' in filename:
    true_logz_lo, true_logz_hi = 0, 0
    problemname = 'asymgauss'
elif 'loggamma' in filename:
    true_logz_lo, true_logz_hi = 0, 0
    problemname = 'loggamma'
else:
    assert False, filename

plt.xscale('log')
plt.ylim(true_logz_lo-2*ndim, true_logz_hi+2*ndim)
xlo, xhi = plt.xlim()
plt.hlines([true_logz_lo, true_logz_hi], xlo, xhi, linestyles=':', color='k')
plt.fill_between([xlo, xhi], [true_logz_lo-ndim/3]*2, [true_logz_hi+ndim/3]*2, color='k', alpha=0.1)
plt.fill_between([xlo, xhi], [true_logz_lo-ndim/9]*2, [true_logz_hi+ndim/9]*2, color='k', alpha=0.1)
plt.xlim(xlo, xhi)
plt.legend(loc='best')
plt.xlabel('number of function calls')
plt.ylabel('$\Delta \log Z$')
print("writing to '%s_%dd_comparison.pdf'" % (problemname, ndim))
plt.savefig(problemname + '_%dd_comparison.pdf' % ndim, bbox_inches='tight')
plt.close()

