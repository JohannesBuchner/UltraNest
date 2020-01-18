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
    method_data.append([ndim, logz, logzerr])
    methods[method] = method_data
    print(ncall, logz, path)

for method, data in sorted(methods.items()):
    x, y, yerr = np.transpose(data)
    i = np.argsort(x)
    x, y, yerr = x[i], y[i], yerr[i]
    if 'rosen' in filename:
        y = y - (3.7 + x * -5.7)
    print(method, x, y)
    plt.errorbar(x=x, y=y, yerr=yerr, marker='o', mfc='w', label=method)

if 'rosen' in filename:
    problemname = 'rosenbrock'
#    x = np.arange(50)
#    plt.plot(x, 3.7 + x*-5.7, '--', color='k')
#    plt.plot(x, 3.7 + x*-5.9, '--', color='k')
#    plt.plot(x, 3.7 + x*-5.5, '--', color='k')
elif 'multishell' in filename:
    problemname = 'multishell'
elif 'asymgauss' in filename:
    problemname = 'asymgauss'
elif 'loggamma' in filename:
    problemname = 'loggamma'
else:
    assert False, filename

#plt.xscale('log')
plt.legend(loc='best')
plt.ylabel('Number of model evaluations')
plt.xlabel('Dimensionality')
print("writing to '%s_scaling_logz.pdf'" % (problemname))
plt.savefig(problemname + '_scaling_logz.pdf', bbox_inches='tight')
plt.close()

