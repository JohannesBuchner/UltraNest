"""Plotting utilities."""

from __future__ import (print_function, division)
from six.moves import range

import logging
import types
import warnings

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator, NullLocator
# from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter

import scipy.stats
import matplotlib.pyplot as plt
import numpy

from .utils import resample_equal
from .utils import quantile as _quantile

try:
    str_type = types.StringTypes
    float_type = types.FloatType
    int_type = types.IntType
except Exception:
    str_type = str
    float_type = float
    int_type = int

import corner

__all__ = ["runplot", "cornerplot", "traceplot", "PredictionBand"]


def cornerplot(results, logger=None):
    """Make a corner plot with corner."""
    paramnames = results['paramnames']
    data = np.array(results['weighted_samples']['points'])
    weights = np.array(results['weighted_samples']['weights'])
    cumsumweights = np.cumsum(weights)

    mask = cumsumweights > 1e-4

    if mask.sum() == 1:
        if logger is not None:
            warn = 'Posterior is still concentrated in a single point:'
            for i, p in enumerate(paramnames):
                v = results['samples'][mask,i]
                warn += "\n" + '    %-20s: %s' % (p, v)

            logger.warning(warn)
            logger.info('Try running longer.')
        return

    # monkey patch to disable a useless warning
    oldfunc = logging.warning
    logging.warning = lambda *args, **kwargs: None
    corner.corner(data[mask,:], weights=weights[mask],
                  labels=paramnames, show_titles=True, quiet=True)
    logging.warning = oldfunc


class PredictionBand(object):
    """Plot bands of model predictions as calculated from a chain.

    call add(y) to add predictions from each chain point

    Example::

        x = numpy.linspace(0, 1, 100)
        band = PredictionBand(x)
        for c in chain:
            band.add(c[0] * x + c[1])
        # add median line
        band.line(color='k')
        # add 1 sigma quantile
        band.shade(color='k', alpha=0.3)
        # add wider quantile
        band.shade(q=0.01, color='gray', alpha=0.1)
        plt.show()

    Parameters
    ----------
    x: array
        The independent variable

    """

    def __init__(self, x, shadeargs={}, lineargs={}):
        """Initialise with independent variable *x*."""
        self.x = x
        self.ys = []
        self.shadeargs = shadeargs
        self.lineargs = lineargs

    def add(self, y):
        """Add a possible prediction *y*."""
        self.ys.append(y)

    def set_shadeargs(self, **kwargs):
        """Set matplotlib style for shading."""
        self.shadeargs = kwargs

    def set_lineargs(self, **kwargs):
        """Set matplotlib style for line."""
        self.lineargs = kwargs

    def get_line(self, q=0.5):
        """Over prediction space x, get quantile *q*. Default is median."""
        if not 0 <= q <= 1:
            raise ValueError("quantile q must be between 0 and 1, not %s" % q)
        assert len(self.ys) > 0, self.ys
        return scipy.stats.mstats.mquantiles(self.ys, q, axis=0)[0]

    def shade(self, q=0.341, **kwargs):
        """Plot a shaded region between 0.5-q and 0.5+q. Default is 1 sigma."""
        if not 0 <= q <= 0.5:
            raise ValueError("quantile distance from the median, q, must be between 0 and 0.5, not %s. For a 99% quantile range, use q=0.48." % q)
        shadeargs = dict(self.shadeargs)
        shadeargs.update(kwargs)
        lo = self.get_line(0.5 - q)
        hi = self.get_line(0.5 + q)
        return plt.fill_between(self.x, lo, hi, **shadeargs)

    def line(self, **kwargs):
        """Plot the median curve."""
        lineargs = dict(self.lineargs)
        lineargs.update(kwargs)
        mid = self.get_line(0.5)
        return plt.plot(self.x, mid, **lineargs)


# the following function is taken from https://github.com/joshspeagle/dynesty/blob/master/dynesty/plotting.py
# Copyright (c) 2017 - Present: Josh Speagle and contributors.
# Copyright (c) 2014 - 2017: Kyle Barbary and contributors.
# https://github.com/joshspeagle/dynesty/blob/master/LICENSE
def runplot(results, span=None, logplot=False, kde=True, nkde=1000,
            color='blue', plot_kwargs=None, label_kwargs=None, lnz_error=True,
            lnz_truth=None, truth_color='red', truth_kwargs=None,
            max_x_ticks=8, max_y_ticks=3, use_math_text=True,
            mark_final_live=True, fig=None
            ):
    """Plot live points, ln(likelihood), ln(weight), and ln(evidence) vs. ln(prior volume).

    Parameters
    ----------
    results : dynesty.results.Results instance
        dynesty.results.Results instance from a nested
        sampling run.
    span : iterable with shape (4,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds *or* a float from `(0., 1.]` giving the
        fraction below the maximum. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.001, 0.2, (5., 6.)]

        Default is `(0., 1.05 * max(data))` for each element.
    logplot : bool, optional
        Whether to plot the evidence on a log scale. Default is `False`.
    kde : bool, optional
        Whether to use kernel density estimation to estimate and plot
        the PDF of the importance weights as a function of log-volume
        (as opposed to the importance weights themselves). Default is
        `True`.
    nkde : int, optional
        The number of grid points used when plotting the kernel density
        estimate. Default is `1000`.
    color : str or iterable with shape (4,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the lines in each subplot.
        Default is `'blue'`.
    plot_kwargs : dict, optional
        Extra keyword arguments that will be passed to `plot`.
    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.
    lnz_error : bool, optional
        Whether to plot the 1, 2, and 3-sigma approximate error bars
        derived from the ln(evidence) error approximation over the course
        of the run. Default is True.
    lnz_truth : float, optional
        A reference value for the evidence that will be overplotted on the
        evidence subplot if provided.
    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color used when plotting `lnz_truth`.
        Default is `'red'`.
    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting
        `lnz_truth`.
    max_x_ticks : int, optional
        Maximum number of ticks allowed for the x axis. Default is `8`.
    max_y_ticks : int, optional
        Maximum number of ticks allowed for the y axis. Default is `4`.
    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.
    mark_final_live : bool, optional
        Whether to indicate the final addition of recycled live points
        (if they were added to the resulting samples) using
        a dashed vertical line. Default is `True`.
    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the run onto the provided figure.
        Otherwise, by default an internal figure is generated.

    Returns
    -------
    runplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output summary plot.

    """
    # Initialize values.
    if label_kwargs is None:
        label_kwargs = dict()
    if plot_kwargs is None:
        plot_kwargs = dict()
    if truth_kwargs is None:
        truth_kwargs = dict()

    # Set defaults.
    plot_kwargs['linewidth'] = plot_kwargs.get('linewidth', 5)
    plot_kwargs['alpha'] = plot_kwargs.get('alpha', 0.7)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 3)

    # Extract results.
    niter = results['niter']  # number of iterations
    logvol = results['logvol']  # ln(prior volume)
    logl = results['logl'] - max(results['logl'])  # ln(normalized likelihood)
    logwt = results['logwt'] - results['logz'][-1]  # ln(importance weight)
    logz = results['logz']  # ln(evidence)
    logzerr = results['logzerr']  # error in ln(evidence)
    weights = results['weights']
    logzerr[~np.isfinite(logzerr)] = 0.
    nsamps = len(logwt)  # number of samples

    # Check whether the run was "static" or "dynamic".
    try:
        nlive = results['samples_n']
        mark_final_live = False
    except Exception:
        nlive = np.ones(niter) * results['nlive']
        if nsamps - niter == results['nlive']:
            nlive_final = np.arange(1, results['nlive'] + 1)[::-1]
            nlive = np.append(nlive, nlive_final)

    # Check if the final set of live points were added to the results.
    if mark_final_live:
        if nsamps - niter == results['nlive']:
            live_idx = niter
        else:
            warnings.warn("The number of iterations and samples differ "
                          "by an amount that isn't the number of final "
                          "live points. `mark_final_live` has been disabled.")
            mark_final_live = False

    # Determine plotting bounds for each subplot.
    data = [nlive, np.exp(logl), weights, logz if logplot else np.exp(logz)]

    kde = kde and (weights * len(logvol) > 0.1).sum() > 10
    if kde:
        try:
            # from scipy.ndimage import gaussian_filter as norm_kde
            from scipy.stats import gaussian_kde
            # Derive kernel density estimate.
            wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
            logvol_new = np.linspace(logvol[0], logvol[-1], nkde)  # resample
            data[2] = wt_kde.pdf(-logvol_new)  # evaluate KDE PDF
        except ImportError:
            kde = False

    if span is None:
        span = [(0., 1.05 * max(d)) for d in data]
        no_span = True
    else:
        no_span = False
    span = list(span)
    if len(span) != 4:
        raise ValueError("More bounds provided in `span` than subplots!")
    for i, _ in enumerate(span):
        try:
            ymin, ymax = span[i]
        except:
            span[i] = (max(data[i]) * span[i], max(data[i]))
    if lnz_error and no_span:
        if logplot:
            zspan = (logz[-1] - 10.3 * 3. * logzerr[-1],
                     logz[-1] + 1.3 * 3. * logzerr[-1])
        else:
            zspan = (0., 1.05 * np.exp(logz[-1] + 3. * logzerr[-1]))
        span[3] = zspan

    # Setting up default plot layout.
    if fig is None:
        fig, axes = pl.subplots(4, 1, figsize=(16, 16))
        xspan = [(0., -min(logvol)) for _ax in axes]
        yspan = span
    else:
        fig, axes = fig
        try:
            axes.reshape(4, 1)
        except:
            raise ValueError("Provided axes do not match the required shape "
                             "for plotting samples.")
        # If figure is provided, keep previous bounds if they were larger.
        xspan = [ax.get_xlim() for ax in axes]
        yspan = [ax.get_ylim() for ax in axes]
        # One exception: if the bounds are the plotting default `(0., 1.)`,
        # overwrite them.
        xspan = [t if t != (0., 1.) else (None, None) for t in xspan]
        yspan = [t if t != (0., 1.) else (None, None) for t in yspan]

    # Set up bounds for plotting.
    for i in range(4):
        if xspan[i][0] is None:
            xmin = None
        else:
            xmin = min(0., xspan[i][0])
        if xspan[i][1] is None:
            xmax = -min(logvol)
        else:
            xmax = max(-min(logvol), xspan[i][1])
        if yspan[i][0] is None:
            ymin = None
        else:
            ymin = min(span[i][0], yspan[i][0])
        if yspan[i][1] is None:
            ymax = span[i][1]
        else:
            ymax = max(span[i][1], yspan[i][1])
        axes[i].set_xlim([xmin, xmax])
        axes[i].set_ylim([ymin, ymax])

    # Plotting.
    labels = ['Live Points', 'Likelihood\n(normalized)',
              'Importance\nWeight', 'Evidence']
    if kde:
        labels[2] += ' PDF'

    for i, d in enumerate(data):

        # Establish axes.
        ax = axes[i]
        # Set color(s)/colormap(s).
        if isinstance(color, str_type):
            c = color
        else:
            c = color[i]
        # Setup axes.
        if max_x_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_x_ticks))
        if max_y_ticks == 0:
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(max_y_ticks))
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.yaxis.set_major_formatter(sf)
        ax.set_xlabel(r"$-\ln X$", **label_kwargs)
        ax.set_ylabel(labels[i], **label_kwargs)
        # Plot run.
        if logplot and i == 3:
            ax.plot(-logvol, d, color=c, **plot_kwargs)
            yspan = [ax.get_ylim() for _ax in axes]
        elif kde and i == 2:
            ax.plot(-logvol_new, d, color=c, **plot_kwargs)
        else:
            ax.plot(-logvol, d, color=c, **plot_kwargs)
        if i == 3 and lnz_error:
            if logplot:
                mask = logz >= ax.get_ylim()[0] - 10
                [ax.fill_between(-logvol[mask], (logz + s * logzerr)[mask],
                                 (logz - s * logzerr)[mask],
                                 color=c, alpha=0.2)
                 for s in range(1, 4)]
            else:
                [ax.fill_between(-logvol, np.exp(logz + s * logzerr),
                                 np.exp(logz - s * logzerr), color=c, alpha=0.2)
                 for s in range(1, 4)]
        # Mark addition of final live points.
        if mark_final_live:
            ax.axvline(-logvol[live_idx], color=c, ls="dashed", lw=2,
                       **plot_kwargs)
            if i == 0:
                ax.axhline(live_idx, color=c, ls="dashed", lw=2,
                           **plot_kwargs)
        # Add truth value(s).
        if i == 3 and lnz_truth is not None:
            if logplot:
                ax.axhline(lnz_truth, color=truth_color, **truth_kwargs)
            else:
                ax.axhline(np.exp(lnz_truth), color=truth_color, **truth_kwargs)

    return fig, axes


def traceplot(results, span=None, quantiles=[0.025, 0.5, 0.975], smooth=0.02,
              post_color='blue', post_kwargs=None, kde=True, nkde=1000,
              trace_cmap='plasma', trace_color=None, trace_kwargs=None,
              connect=False, connect_highlight=10, connect_color='red',
              connect_kwargs=None, max_n_ticks=5, use_math_text=False,
              labels=None, label_kwargs=None,
              show_titles=False, title_fmt=".2f", title_kwargs=None,
              truths=None, truth_color='red', truth_kwargs=None,
              verbose=False, fig=None):
    """Plot traces and marginalized posteriors for each parameter.

    Parameters
    ----------
    results : `~dynesty.results.Results` instance
        A `~dynesty.results.Results` instance from a nested
        sampling run. **Compatible with results derived from**
        `nestle <http://kylebarbary.com/nestle/>`_.
    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval) for each
        parameter.
    quantiles : iterable, optional
        A list of fractional quantiles to overplot on the 1-D marginalized
        posteriors as vertical dashed lines. Default is `[0.025, 0.5, 0.975]`
        (the 95%/2-sigma credible interval).
    smooth : float or iterable with shape (ndim,), optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 1-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.02` (2% smoothing). If an integer is provided instead,
        this will instead default to a simple (weighted) histogram with
        `bins=smooth`.
    post_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the histograms.
        Default is `'blue'`.
    post_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the
        marginalized 1-D posteriors.
    kde : bool, optional
        Whether to use kernel density estimation to estimate and plot
        the PDF of the importance weights as a function of log-volume
        (as opposed to the importance weights themselves). Default is
        `True`.
    nkde : int, optional
        The number of grid points used when plotting the kernel density
        estimate. Default is `1000`.
    trace_cmap : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style colormap (either a single colormap or a
        different colormap for each subplot) used when plotting the traces,
        where each point is colored according to its weight. Default is
        `'plasma'`.
    trace_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a
        different color for each subplot) used when plotting the traces.
        This overrides the `trace_cmap` option by giving all points
        the same color. Default is `None` (not used).
    trace_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the traces.
    connect : bool, optional
        Whether to draw lines connecting the paths of unique particles.
        Default is `False`.
    connect_highlight : int or iterable, optional
        If `connect=True`, highlights the paths of a specific set of
        particles. If an integer is passed, :data:`connect_highlight`
        random particle paths will be highlighted. If an iterable is passed,
        then the particle paths corresponding to the provided indices
        will be highlighted.
    connect_color : str, optional
        The color of the highlighted particle paths. Default is `'red'`.
    connect_kwargs : dict, optional
        Extra keyword arguments used for plotting particle paths.
    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.
    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.
    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.
    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.
    show_titles : bool, optional
        Whether to display a title above each 1-D marginalized posterior
        showing the 0.5 quantile along with the upper/lower bounds associated
        with the 0.025 and 0.975 (95%/2-sigma credible interval) quantiles.
        Default is `True`.
    title_fmt : str, optional
        The format string for the quantiles provided in the title. Default is
        `'.2f'`.
    title_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_title` command.
    truths : iterable with shape (ndim,), optional
        A list of reference values that will be overplotted on the traces and
        marginalized 1-D posteriors as solid horizontal/vertical lines.
        Individual values can be exempt using `None`. Default is `None`.
    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting `truths`.
        Default is `'red'`.
    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the vertical
        and horizontal lines with `truths`.
    verbose : bool, optional
        Whether to print the values of the computed quantiles associated with
        each parameter. Default is `False`.
    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.

    Returns
    -------
    traceplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output trace plot.

    """
    # Initialize values.
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if trace_kwargs is None:
        trace_kwargs = dict()
    if connect_kwargs is None:
        connect_kwargs = dict()
    if post_kwargs is None:
        post_kwargs = dict()
    if truth_kwargs is None:
        truth_kwargs = dict()

    # Set defaults.
    connect_kwargs['alpha'] = connect_kwargs.get('alpha', 0.7)
    post_kwargs['alpha'] = post_kwargs.get('alpha', 0.6)
    trace_kwargs['s'] = trace_kwargs.get('s', 3)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)

    # Extract weighted samples.
    samples = results['samples']
    logvol = results['logvol']
    weights = results['weights']

    wts = weights
    kde = kde and (weights * len(logvol) > 0.1).sum() > 10
    if kde:
        try:
            from scipy.ndimage import gaussian_filter as norm_kde
            from scipy.stats import gaussian_kde
            # Derive kernel density estimate.
            wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
            logvol_grid = np.linspace(logvol[0], logvol[-1], nkde)  # resample
            wt_grid = wt_kde.pdf(-logvol_grid)  # evaluate KDE PDF
            wts = np.interp(-logvol, -logvol_grid, wt_grid)  # interpolate
        except ImportError:
            kde = False

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more dimensions than samples!"
    ndim, nsamps = samples.shape

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Check ln(volume).
    if logvol.ndim != 1:
        raise ValueError("Ln(volume)'s must be 1-D.")
    if nsamps != logvol.shape[0]:
        raise ValueError("The number of ln(volume)'s and samples disagree!")

    # Check sample IDs.
    if connect:
        try:
            samples_id = results['samples_id']
            uid = np.unique(samples_id)
        except:
            raise ValueError("Sample IDs are not defined!")
        try:
            ids = connect_highlight[0]
            ids = connect_highlight
        except:
            ids = np.random.choice(uid, size=connect_highlight, replace=False)

    # Determine plotting bounds for marginalized 1-D posteriors.
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)

    # Setting up labels.
    if labels is None:
        labels = [r"$x_{%d}$" % (i + 1) for i in range(ndim)]

    # Setting up smoothing.
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth for i in range(ndim)]

    # Setting up default plot layout.
    if fig is None:
        fig, axes = pl.subplots(ndim, 2, figsize=(12, 3 * ndim))
    else:
        fig, axes = fig
        try:
            axes.reshape(ndim, 2)
        except:
            raise ValueError("Provided axes do not match the required shape "
                             "for plotting samples.")

    # Plotting.
    for i, x in enumerate(samples):

        # Plot trace.

        # Establish axes.
        if np.shape(samples)[0] == 1:
            ax = axes[1]
        else:
            ax = axes[i, 0]
        # Set color(s)/colormap(s).
        if trace_color is not None:
            if isinstance(trace_color, str_type):
                color = trace_color
            else:
                color = trace_color[i]
        else:
            color = wts
        if isinstance(trace_cmap, str_type):
            cmap = trace_cmap
        else:
            cmap = trace_cmap[i]
        # Setup axes.
        ax.set_xlim([0., -min(logvol)])
        ax.set_ylim([min(x), max(x)])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks))
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.yaxis.set_major_formatter(sf)
        ax.set_xlabel(r"$-\ln X$", **label_kwargs)
        ax.set_ylabel(labels[i], **label_kwargs)
        # Generate scatter plot.
        ax.scatter(-logvol, x, c=color, cmap=cmap, **trace_kwargs)
        if connect:
            # Add lines highlighting specific particle paths.
            for j in ids:
                sel = (samples_id == j)
                ax.plot(-logvol[sel], x[sel], color=connect_color,
                        **connect_kwargs)
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                [ax.axhline(t, color=truth_color, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axhline(truths[i], color=truth_color, **truth_kwargs)

        # Plot marginalized 1-D posterior.

        # Establish axes.
        if np.shape(samples)[0] == 1:
            ax = axes[0]
        else:
            ax = axes[i, 1]
        # Set color(s).
        if isinstance(post_color, str_type):
            color = post_color
        else:
            color = post_color[i]
        # Setup axes
        ax.set_xlim(span[i])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(NullLocator())
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        ax.set_xlabel(labels[i], **label_kwargs)
        # Generate distribution.
        s = smooth[i]
        if isinstance(s, int_type):
            # If `s` is an integer, plot a weighted histogram with
            # `s` bins within the provided bounds.
            n, b, _ = ax.hist(x, bins=s, weights=weights, color=color,
                              range=np.sort(span[i]), **post_kwargs)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            if kde:
                bins = int(round(10. / s))
                n, b = np.histogram(x, bins=bins, weights=weights,
                                    range=np.sort(span[i]))
                x0 = 0.5 * (b[1:] + b[:-1])
                n = norm_kde(n, 10.)
                y0 = n
                ax.fill_between(x0, y0, color=color, **post_kwargs)
            else:
                bins = 40
                n, b = np.histogram(x, bins=bins, weights=weights,
                                    range=np.sort(span[i]))
                x0 = 0.5 * (b[1:] + b[:-1])
                y0 = n
                ax.fill_between(x0, y0, color=color, **post_kwargs)
        ax.set_ylim([0., max(y0) * 1.05])
        # Plot quantiles.
        if quantiles is not None and len(quantiles) > 0:
            qs = _quantile(x, quantiles, weights=weights)
            for q in qs:
                ax.axvline(q, lw=2, ls="dashed", color=color)
            if verbose:
                print("Quantiles:")
                print(labels[i], [blob for blob in zip(quantiles, qs)])
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                [ax.axvline(t, color=truth_color, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axvline(truths[i], color=truth_color, **truth_kwargs)
        # Set titles.
        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = _quantile(x, [0.025, 0.5, 0.975], weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, **title_kwargs)

    return fig, axes
