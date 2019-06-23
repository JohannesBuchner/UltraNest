"""
Visual impression of current exploration
"""

from __future__ import print_function, division

import sys
import shutil


from numpy import log10
import numpy as np
import string
clusteridstrings = ['%d' % i for i in range(10)] + list(string.ascii_uppercase) + list(string.ascii_lowercase)

spearman = None
try:
    import scipy.stats
    spearman = scipy.stats.spearmanr
except ImportError:
    pass


def nicelogger(points, info, region, transformLayer, region_fresh=False):
    #u, p, logl = points['u'], points['p'], points['logl']
    p = points['p']
    paramnames = info['paramnames']
    #print()
    #print('lnZ = %.1f, remainder = %.1f, lnLike = %.1f | Efficiency: %d/%d = %.4f%%\r' % (
    #      logz, logz_remain, np.max(logl), ncall, it, it * 100 / ncall))
    
    plo = p.min(axis=0)
    phi = p.max(axis=0)
    expos = log10(np.abs([plo, phi]))
    expolo = np.floor(np.min(expos, axis=0))
    expohi = np.ceil(np.max(expos, axis=0))
    is_negative = plo < 0
    plo_rounded = np.where(is_negative, -10**expohi, 10**expolo)
    phi_rounded = np.where(is_negative,  10**expohi, 10**expohi)

    if sys.stderr.isatty() and hasattr(shutil, 'get_terminal_size'):
        columns, _rows = shutil.get_terminal_size(fallback=(80, 25))
    else:
        columns, _rows = 80, 25

    paramwidth = max([len(pname) for pname in paramnames])
    width = columns - 23 - paramwidth
    width = max(width, 10)
    indices = ((p - plo_rounded) * width / (phi_rounded - plo_rounded).reshape((1, -1))).astype(int)
    indices[indices >= width] = width - 1
    indices[indices < 0] = 0
    ndim = len(plo)
    
    print()
    print()
    clusterids = transformLayer.clusterids % len(clusteridstrings)
    nmodes = transformLayer.nclusters
    print("Mono-modal" if nmodes == 1 else "Have %d modes" % nmodes, 
        "Volume: ~%.2e" % region.estimate_volume(), '*' if region_fresh else ' ',
        "Expected Volume: %.2e" % np.exp(info['logvol']))
    
    if ndim == 1:
        pass
    elif ndim == 2 and spearman is not None:
        rho, pval = spearman(p)
        if pval < 0.01 and abs(rho) > 0.75:
            print("   %s between %s and %s: rho=%.2f" % (
                'positive degeneracy' if rho > 0 else 'negative degeneracy',
                paramnames[0], paramnames[1], rho))
    elif spearman is not None:
        rho, pval = spearman(p)
        for i, param in enumerate(paramnames):
            for j, param2 in enumerate(paramnames[:i]):
                if pval[i,j] < 0.01 and abs(rho[i,j]) > 0.99:
                    print("   perfect %s between %s and %s" % (
                        'positive relation' if rho[i,j] > 0 else 'negative relation',
                        param, param2))
                elif pval[i,j] < 0.01 and abs(rho[i,j]) > 0.75:
                    print("   %s between %s and %s: rho=%.2f" % (
                        'positive degeneracy' if rho[i,j] > 0 else 'negative degeneracy',
                        param, param2, rho[i,j]))
    print()
    
    for i, param in enumerate(paramnames):
        if nmodes == 1:
            line = [' ' for i in range(width)]
            for j in np.unique(indices[:,i]):
                line[j] = '*'
            linestr = ''.join(line)
        else:
            line = [' ' for i in range(width)]
            for clusterid, j in zip(clusterids, indices[:,i]):
                if clusterid > 0 and line[j] in (' ', '0'):
                    # set it to correct cluster id
                    line[j] = clusteridstrings[clusterid]
                elif clusterid == 0 and line[j] == ' ':
                    # empty, so set it although we don't know the cluster id
                    line[j] = '0'
                #else:
                #    line[j] = '*'
            linestr = ''.join(line)
        
        fmt = '%+.1e'
        if -1 <= expolo[i] <= 2 and -1 <= expohi[i] <= 2:
            if not is_negative[i]:
                plo_rounded[i] = 0
            fmt = '%+.1f'
        if -4 <= expolo[i] <= 0 and -4 <= expohi[i] <= 0:
            fmt = '%%+.%df' % (-min(expolo[i], expohi[i]))
        
        line = linestr
        ilo, ihi = indices[:,i].min(), indices[:,i].max()
        if ilo > 10:
            assert line[:10] == ' '*10
            leftstr = fmt % plo[i]
            j = ilo - 2 - len(leftstr) # left-bound
            if j < width and j > 0:
                line = line[:j] + leftstr + line[j + len(leftstr):]
        if ihi < width - 10:
            rightstr = fmt % phi[i]
            j = ihi + 3 # right-bound
            if j < width and j > 0:
                line = line[:j] + rightstr + line[j + len(rightstr):]

			
        parampadded = ('%%-%ds' % paramwidth) % param
        print('%s: %09s|%s|%9s' % (parampadded, fmt % plo_rounded[i], line, fmt % phi_rounded[i]))
    
    print()
    
