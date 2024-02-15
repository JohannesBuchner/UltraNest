import numpy as np
import tempfile
import os
from ultranest.plot import PredictionBand
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
