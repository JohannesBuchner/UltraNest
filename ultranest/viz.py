"""Visual impression of current exploration."""

from __future__ import print_function, division

import sys
import shutil
from numpy import log10
import numpy as np
import string
from xml.sax.saxutils import escape as html_escape


clusteridstrings = ['%d' % i for i in range(10)] + list(string.ascii_uppercase) + list(string.ascii_lowercase)

spearman = None
try:
    import scipy.stats
    spearman = scipy.stats.spearmanr
except ImportError:
    pass


def round_parameterlimits(plo, phi, paramlimitguess=None):
    """Guess the current parameter range.

    Parameters
    -----------
    plo: array of floats
        for each parameter, current minimum value
    phi: array of floats
        for each parameter, current maximum value
    paramlimitguess: array of float tuples
        for each parameter, guess of parameter range if available

    Returns
    -------
    plo_rounded: array of floats
        for each parameter, rounded minimum value
    phi_rounded: array of floats
        for each parameter, rounded maximum value
    formats: array of float tuples
        for each parameter, string format for representing it.

    """
    with np.errstate(divide='ignore'):
        expos = log10(np.abs([plo, phi]))
    expolo = np.floor(np.min(expos, axis=0))
    expohi = np.ceil(np.max(expos, axis=0))
    is_negative = plo < 0
    plo_rounded = np.where(is_negative, -10**expohi, 0)
    phi_rounded = np.where(is_negative, 10**expohi, 10**expohi)

    if paramlimitguess is not None:
        for i, (plo_guess, phi_guess) in enumerate(paramlimitguess):
            # if plo_guess is higher than what we thought, we can increase to match
            if plo_guess <= plo[i] and plo_guess >= plo_rounded[i]:
                plo_rounded[i] = plo_guess
            if phi_guess >= phi[i] and phi_guess <= phi_rounded[i]:
                phi_rounded[i] = phi_guess

    formats = []
    for i in range(len(plo)):
        fmt = '%+.1e'
        if -1 <= expolo[i] <= 2 and -1 <= expohi[i] <= 2:
            fmt = '%+.1f'
        if -4 <= expolo[i] <= 0 and -4 <= expohi[i] <= 0:
            fmt = '%%+.%df' % (max(0, -min(expolo[i], expohi[i])))
        if phi[i] == plo[i]:
            fmt = '%+.1f'
        elif fmt % plo[i] == fmt % phi[i]:
            fmt = '%%+.%df' % (max(0, -int(np.floor(log10(phi[i] - plo[i])))))
        formats.append(fmt)

    return plo_rounded, phi_rounded, formats


def nicelogger(points, info, region, transformLayer, region_fresh=False):
    """Log current live points and integration progress to stdout.

    Parameters
    -----------
    points: dict with keys "u", "p", "logl"
        live points (u: cube coordinates, p: transformed coordinates,
        logl: loglikelihood values)
    info: dict
        integration information. Keys are:

        - paramlims (optional): parameter ranges
        - logvol: expected volume at this iteration

    region: MLFriends
        Current region.
    transformLayer: ScaleLayer or AffineLayer
        Current transformLayer (for clustering information).
    region_fresh: bool
        Whether the region was just updated.

    """
    p = points['p']
    paramnames = info['paramnames']
    # print()
    # print('lnZ = %.1f, remainder = %.1f, lnLike = %.1f | Efficiency: %d/%d = %.4f%%\r' % (
    #       logz, logz_remain, np.max(logl), ncall, it, it * 100 / ncall))

    plo = p.min(axis=0)
    phi = p.max(axis=0)
    plo_rounded, phi_rounded, paramformats = round_parameterlimits(plo, phi, paramlimitguess=info.get('paramlims'))

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
    print(
        "Mono-modal" if nmodes == 1 else "Have %d modes" % nmodes,
        "Volume: ~exp(%.2f)" % region.estimate_volume(), '*' if region_fresh else ' ',
        "Expected Volume: exp(%.2f)" % info['logvol'],
        '' if 'order_test_correlation' not in info else
        ("Quality: correlation length: %d (%s)" % (info['order_test_correlation'], '+' if info['order_test_direction'] >= 0 else '-'))
        if np.isfinite(info['order_test_correlation']) else "Quality: ok",
    )

    print()
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
        if np.isfinite(pval).all() and pval.ndim == 2:
            for i, param in enumerate(paramnames):
                for j, param2 in enumerate(paramnames[:i]):
                    if pval[i,j] < 0.01 and abs(rho[i,j]) > 0.99:
                        s = 'positive relation' if rho[i,j] > 0 else 'negative relation'
                        print("   perfect %s between %s and %s" % (s, param, param2))
                    elif pval[i,j] < 0.01 and abs(rho[i,j]) > 0.75:
                        s = 'positive degeneracy' if rho[i,j] > 0 else 'negative degeneracy'
                        print("   %s between %s and %s: rho=%.2f" % (s, param, param2, rho[i,j]))

    for i, (param, fmt) in enumerate(zip(paramnames, paramformats)):
        if nmodes == 1:
            line = [' ' for _ in range(width)]
            for j in np.unique(indices[:,i]):
                line[j] = '*'
            linestr = ''.join(line)
        else:
            line = [' ' for _ in range(width)]
            for clusterid, j in zip(clusterids, indices[:,i]):
                if clusterid > 0 and line[j] in (' ', '0'):
                    # set it to correct cluster id
                    line[j] = clusteridstrings[clusterid]
                elif clusterid == 0 and line[j] == ' ':
                    # empty, so set it although we don't know the cluster id
                    line[j] = '0'
                # else:
                #    line[j] = '*'
            linestr = ''.join(line)

        line = linestr
        ilo, ihi = indices[:,i].min(), indices[:,i].max()
        if ilo > 10:
            assert line[:10] == ' ' * 10
            leftstr = fmt % plo[i]
            j = ilo - 2 - len(leftstr)  # left-bound
            if j < width and j > 0:
                line = line[:j] + leftstr + line[j + len(leftstr):]
        if ihi < width - 10:
            rightstr = fmt % phi[i]
            j = ihi + 3  # right-bound
            if j < width and j > 0:
                line = line[:j] + rightstr + line[j + len(rightstr):]

        parampadded = ('%%-%ds' % paramwidth) % param
        print('%s: %09s|%s|%9s' % (parampadded, fmt % plo_rounded[i], line, fmt % phi_rounded[i]))

    print()


def isnotebook():
    """Check if running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class LivePointsWidget(object):
    """
    Widget for ipython and jupyter notebooks.

    Shows where the live points are currently in parameter space.
    """

    def __init__(self):
        """Initialise. To draw, call .initialize()."""
        self.grid = None
        self.label = None
        self.laststatus = None

    def initialize(self, paramnames, width):
        """Set up and display widget.

        Parameters
        ----------
        paramnames: list of str
            Parameter names
        width: int
            number of html table columns.

        """
        from ipywidgets import HTML, VBox, Layout, GridspecLayout
        from IPython.display import display

        grid = GridspecLayout(len(paramnames), width + 3)
        self.laststatus = []
        for a, paramname in enumerate(paramnames):
            self.laststatus.append('*' * width)
            htmlcode = "<div style='background-color:#6E6BF4;'>&nbsp;</div>"
            for b in range(width):
                grid[a, b + 2] = HTML(htmlcode, layout=Layout(margin="0"))
            htmlcode = "<div style='background-color:#FFB858; font-weight:bold; padding-right: 2em;'>%s</div>"
            grid[a, 0] = HTML(htmlcode % html_escape(paramname), layout=Layout(margin="0"))
            grid[a, 1] = HTML("...", layout=Layout(margin="0"))
            grid[a,-1] = HTML("...", layout=Layout(margin="0"))
        self.grid = grid

        self.label = HTML()
        box = VBox(children=[self.label, grid])
        display(box)

    def __call__(self, points, info, region, transformLayer, region_fresh=False):
        """Update widget to show current live points and integration progress to stdout.

        Parameters
        -----------
        points: dict with keys u, p, logl
            live points (u: cube coordinates, p: transformed coordinates,
            logl: loglikelihood values)
        info: dict
            integration information. Keys are:

            - paramlims (optional): parameter ranges
            - logvol: expected volume at this iteration

        region: MLFriends
            Current region.
        transformLayer: ScaleLayer or AffineLayer
            Current transformLayer (for clustering information).
        region_fresh: bool
            Whether the region was just updated.

        """
        # t = time.time()
        # if self.lastupdate is not None and self.lastupdate < t - 5:
        #    return
        # self.lastupdate = t
        # u, p, logl = points['u'], points['p'], points['logl']
        p = points['p']
        paramnames = info['paramnames']
        # print()
        # print('lnZ = %.1f, remainder = %.1f, lnLike = %.1f | Efficiency: %d/%d = %.4f%%\r' % (
        #       logz, logz_remain, np.max(logl), ncall, it, it * 100 / ncall))

        plo = p.min(axis=0)
        phi = p.max(axis=0)
        plo_rounded, phi_rounded, paramformats = round_parameterlimits(plo, phi, paramlimitguess=info.get('paramlims'))

        width = 50

        if self.grid is None:
            self.initialize(paramnames, width)

        with np.errstate(invalid="ignore"):
            indices = ((p - plo_rounded) * width / (phi_rounded - plo_rounded).reshape((1, -1))).astype(int)
        indices[indices >= width] = width - 1
        indices[indices < 0] = 0
        ndim = len(plo)

        clusterids = transformLayer.clusterids % len(clusteridstrings)
        nmodes = transformLayer.nclusters
        labeltext = ("Mono-modal" if nmodes == 1 else "Have %d modes" % nmodes) + \
            (" | Volume: ~exp(%.2f) " % region.estimate_volume()) + ('*' if region_fresh else ' ') + \
            " | Expected Volume: exp(%.2f)" % info['logvol'] + \
            ('' if 'order_test_correlation' not in info else
             (" | Quality: correlation length: %d (%s)" % (info['order_test_correlation'], '+' if info['order_test_direction'] >= 0 else '-'))
             if np.isfinite(info['order_test_correlation']) else " | Quality: ok")

        if ndim == 1:
            pass
        elif ndim == 2 and spearman is not None:
            rho, pval = spearman(p)
            if pval < 0.01 and abs(rho) > 0.75:
                labeltext += ("<br/>   %s between %s and %s: rho=%.2f" % (
                    'positive degeneracy' if rho > 0 else 'negative degeneracy',
                    paramnames[0], paramnames[1], rho))
        elif spearman is not None:
            rho, pval = spearman(p)
            for i, param in enumerate(paramnames):
                for j, param2 in enumerate(paramnames[:i]):
                    if pval[i,j] < 0.01 and abs(rho[i,j]) > 0.99:
                        labeltext += ("<br/>   perfect %s between %s and %s" % (
                            'positive relation' if rho[i,j] > 0 else 'negative relation',
                            param2, param))
                    elif pval[i,j] < 0.01 and abs(rho[i,j]) > 0.75:
                        labeltext += ("<br/>   %s between %s and %s: rho=%.2f" % (
                            'positive degeneracy' if rho[i,j] > 0 else 'negative degeneracy',
                            param2, param, rho[i,j]))

        for i, (param, fmt) in enumerate(zip(paramnames, paramformats)):
            if nmodes == 1:
                line = [' ' for _ in range(width)]
                for j in np.unique(indices[:,i]):
                    line[j] = '*'
                linestr = ''.join(line)
            else:
                line = [' ' for _ in range(width)]
                for clusterid, j in zip(clusterids, indices[:,i]):
                    if clusterid > 0 and line[j] in (' ', '0'):
                        # set it to correct cluster id
                        line[j] = clusteridstrings[clusterid]
                    elif clusterid == 0 and line[j] == ' ':
                        # empty, so set it although we don't know the cluster id
                        line[j] = '0'
                    # else:
                    #     line[j] = '*'
                linestr = ''.join(line)

            oldlinestr = self.laststatus[i]
            for j, (c, d) in enumerate(zip(linestr, oldlinestr)):
                if c != d:
                    if c == ' ':
                        self.grid[i, j + 2].value = "<div style='background-color:white;'>&nbsp;</div>"
                    else:
                        self.grid[i, j + 2].value = "<div style='background-color:#6E6BF4; font-family:monospace'>%s</div>" % c.replace('*', '&nbsp;')

            self.laststatus[i] = linestr
            # self.grid[i,0].value = param
            self.grid[i, 1].value = fmt % plo_rounded[i]
            self.grid[i,-1].value = fmt % phi_rounded[i]

        self.label.value = labeltext


def get_default_viz_callback():
    """Get default callback.

    LivePointsWidget for Jupyter notebook, nicelogger otherwise.
    """
    if isnotebook():
        return LivePointsWidget()
    else:
        return nicelogger
