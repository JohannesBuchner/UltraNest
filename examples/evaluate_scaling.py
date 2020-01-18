import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, log
import sys
import json
import scipy.stats, scipy.special

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
    method_data.append([ndim, ncall])
    methods[method] = method_data
    print(ncall, logz, path)

lines = []
for method, data in sorted(methods.items()):
    x, y = np.transpose(data)
    i = np.argsort(x)
    x, y = x[i], y[i]
    print(method, x, y)
    plt.plot(x, y, marker='o', mfc='w', label=method)
    if method == 'harm-move-distance-midway':
        xgrid = np.linspace(x.max() / 20, x.max()*3, 50)
        #plt.plot(xgrid, (xgrid / x.max())**2 * y.max(), '--', color='k', label='quadratic scaling', alpha=0.4)
        lines.append(('quadratic scaling', xgrid, (xgrid / x.max())**2 * y.max(), dict(color='k', ls='--')))
        lines.append(('cubic scaling', xgrid, (xgrid / x.max())**3 * y.max(), dict(color='k', ls='-.')))
        #plt.plot(xgrid, (xgrid / x.max())**3 * y.max(), '--', color='gray', label='cubic scaling', alpha=0.4)

    if method == 'MLFriends':
        #xgrid = np.linspace(x.max() / 3, x.max(), 50)
        #plt.plot(xgrid, (xgrid / xgrid.max())**4 * y.max(), '-', color='k', label='quadratic scaling')
        #xgrid = np.linspace(x.max()/2, x.max(), 50)
        #plt.plot(xgrid, np.exp(xgrid - x.max()) * y.max(), '-', color='k', label='exponential scaling')
        xgrid = np.linspace(x.min(), x.max() * 3, 50)
        lines.append(('exponential', xgrid, np.exp((xgrid - x[-2]) / (x[-1] - x[-2]) * np.log(y[-1] / y[-2])) * y[-2], 
            dict(color='gray', ls=':')))
        #plt.plot(xgrid, np.exp((xgrid - x[-2]) / (x[-1] - x[-2]) * np.log(y[-1] / y[-2])) * y[-2], '--', color='k', label='exponential scaling', alpha=0.4)

if 'rosen' in filename:
    problemname = 'rosenbrock'
elif 'multishell' in filename:
    problemname = 'multishell'
elif 'asymgauss' in filename:
    problemname = 'asymgauss'
elif 'loggamma' in filename:
    problemname = 'loggamma'
else:
    assert False, filename

plt.xscale('log')
plt.yscale('log')
ylo, yhi = plt.ylim()
xlo, xhi = plt.xlim()
plt.ylim(ylo, yhi)
plt.xlim(xlo, xhi)
for name, x, y, opts in lines:
    plt.plot(x, y, label=name, alpha=0.4, **opts)
plt.legend(loc='best')
plt.ylabel('Number of model evaluations')
plt.xlabel('Dimensionality')
print("writing to '%s_scaling.pdf'" % (problemname))
plt.savefig(problemname + '_scaling.pdf', bbox_inches='tight')
plt.close()

