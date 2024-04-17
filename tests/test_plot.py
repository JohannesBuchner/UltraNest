import numpy as np
import tempfile
import os
from ultranest.plot import PredictionBand, highest_density_interval_from_samples
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import pytest

def test_PredictionBand():
    
    import numpy
    chain = numpy.random.uniform(size=(20, 2))


    x = numpy.linspace(0, 1, 100)
    band = PredictionBand(x)
    for c in chain:
        band.add(c[0] * x + c[1])
    # add median line. As an option a matplotlib ax can be given.
    band.line(color='k')
    # add 1 sigma quantile
    band.shade(color='k', alpha=0.3)
    # add wider quantile
    band.shade(q=0.01, color='gray', alpha=0.1)
    plt.savefig('test-predictionband.pdf')
    plt.close()
    
    # add median line. As an option a matplotlib ax can be given.
    fig, (ax1, ax2) = plt.subplots(1, 2)
    band.line(color='k', ax=ax1)
    band.line(color='k', ax=ax2)
    # add 1 sigma quantile
    with pytest.raises(ValueError):
        band.shade(q=0.6, ax=ax1)
    with pytest.raises(ValueError):
        band.shade(q=np.nan, ax=ax2)
    band.shade(q=0.01, color='gray', alpha=0.3, ax=ax1)
    plt.savefig('test-predictionband2.pdf')
    plt.close()


def test_hdi():
    rng = np.random.RandomState(2)
    x = rng.normal(size=100000)
    xmid, xerrlo, xerrhi = highest_density_interval_from_samples(x, xlo=None, xhi=None, probability_level=0.68)
    assert -0.02 < xmid < 0.02
    assert 0.98 < xerrlo < 1.02
    assert 0.98 < xerrhi < 1.02

    xpmid, xperrlo, xperrhi = highest_density_interval_from_samples(np.abs(x), xlo=0, xhi=None, probability_level=0.68)
    assert 0 <= xpmid < 0.02
    assert 0.98 < xperrhi < 1.02
    assert 0 <= xperrlo < 0.02

    xpmid, xperrlo, xperrhi = highest_density_interval_from_samples(-np.abs(x), xlo=None, xhi=0, probability_level=0.68)
    assert -0.02 < xpmid <= 0
    assert 0.98 < xperrlo < 1.02
    assert 0 <= xperrhi < 0.02

    xmid, xerrlo, xerrhi = highest_density_interval_from_samples(x, xlo=None, xhi=None, probability_level=0.955)
    assert -0.02 < xmid < 0.02
    assert 1.98 < xerrlo < 2.02
    assert 1.98 < xerrhi < 2.02

    xpmid, xperrlo, xperrhi = highest_density_interval_from_samples(np.abs(x), xlo=0, xhi=None, probability_level=0.955)
    assert 0 <= xpmid < 0.02
    assert 1.98 < xperrhi < 2.02
    assert 0 <= xperrlo < 0.02

    xpmid, xperrlo, xperrhi = highest_density_interval_from_samples(-np.abs(x), xlo=None, xhi=0, probability_level=0.955)
    assert -0.02 < xpmid <= 0
    assert 1.98 < xperrlo < 2.02
    assert 0 <= xperrhi < 0.02

    u = rng.beta(2, 2, size=100000)
    umid, uerrlo, uerrhi = highest_density_interval_from_samples(u, xlo=0, xhi=1, probability_level=0.68)
    print(umid, uerrlo, uerrhi)
    assert abs(umid - 0.5) < 0.02, umid
    assert abs(uerrlo - 0.25) < 0.02, umid
    assert abs(uerrhi - 0.25) < 0.02, umid

    umid, uerrlo, uerrhi = highest_density_interval_from_samples(u, xlo=None, xhi=None, probability_level=0.68)
    print(umid, uerrlo, uerrhi)
    assert abs(umid - 0.5) < 0.02, umid
    assert abs(uerrlo - 0.25) < 0.02, umid
    assert abs(uerrhi - 0.25) < 0.02, umid
