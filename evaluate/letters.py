from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import time
import numpy as np
import numpy
import scipy.stats
import scipy.interpolate
import tqdm
import matplotlib.pyplot as plt
import joblib
mem = joblib.Memory('.', verbose=False)

def sample_character(char, path='OpenSans-Bold.ttf', fontsize=60, width_per_cell=0.5, num_samples=1000, center_coords=(0,0), manifold_type="e"):

    """
    Based on https://stackoverflow.com/a/27753869/190597 (jsheperd)
    """

    font = ImageFont.truetype(path, fontsize)
    #font = ImageFont.truetype(path, fontsize) 
    #w, h = font.getsize(char)
    #h = int(h * 1.2)
    w, h = 70 * len(char), 70 * len(char)
    image = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, font=font)
    arr = 255.0 - np.asarray(image)
    arr = scipy.ndimage.gaussian_filter(arr, 1, truncate=20)

    return arr.T

#word = 'ULTRANEST'
word = 'II'
characters = [sample_character(c, 'comic') for c in word]

for i, c in enumerate(characters):
    plt.subplot(len(characters), 1, i+1)
    plt.imshow(c.T >= c.max() / 2)
plt.savefig('letters.png', bbox_inches='tight')
plt.close()

startpoint = []
borders = []
interpolators = []
expect_filled = 1
Lmin = 0
for c in characters:
    i, j = np.where(c[:,::-1] >= c.max() / 2)
    xlo, xhi, ylo, yhi = i.min(), i.max(), j.min(), j.max()
    
    # flip x and y, to correct for transposition
    xscale = 1. / c.shape[0]
    yscale = 1. / c.shape[1]
    
    a = (i * xscale - xlo) / (xhi - xlo) * 3
    a[a > 2] = 2
    a[a < 0] = 0
    b = (j * yscale - ylo) / (yhi - ylo) * 3
    b[b > 2] = 2
    b[b < 0] = 0
    # assign each grid a index, compute global index by shifting
    expect_filled *= len(np.unique(a.astype(int) * 3 + b.astype(int)))
    
    startpoint += [i[0] * xscale, j[0] * yscale]
    borders.append((ylo * yscale, yhi * yscale, xlo * xscale, xhi * xscale))
    print('maxval:', c.max(), 'border:', borders[-1], 'start guess:', startpoint[-2:])
    Lmin += c.max() / 10. * 0.4
    interpolators.append(
        scipy.interpolate.RegularGridInterpolator(
            # flip x and y, to correct for transposition
            (np.linspace(0, 1, c.shape[1]), np.linspace(0, 1, c.shape[0])), 
            # reverse y-axis, to correct for transposition
            c[:,::-1] / 10,
            'linear', bounds_error=True)
    )
    print("   ", interpolators[-1]((i[0] * xscale, j[0] * yscale)))
#    plt.subplot(len(characters), 1, len(borders)+1)
#    plt.imshow(c)
#plt.savefig('letters.png', bbox_inches='tight')
#plt.close()

startpoint = np.array(startpoint)
paramnames = ['p%d' % (i+1) for i in range(len(characters)*2)]
nparams = len(paramnames)

def loglikelihood(params):
    like = 0.0
    for i, interpolator in enumerate(interpolators):
        like += interpolator(params[:, i*2:(i+1)*2])
    return like

def transform(x): return x

def flat_indicator(params):
    if (params >= 0).all() and (params <= 1).all():
        like = 0.0
        for i, interpolator in enumerate(interpolators):
            like += interpolator(params[i*2:(i+1)*2])
        if like >= Lmin:
            return 0.
    return -np.inf


def frac_filled(samples):
    key = 0
    for i, (xlo, xhi, ylo, yhi) in enumerate(borders):
        # divide x and y in a 3x3 grid:
        a = (samples[:,i*2] - xlo) / (xhi - xlo) * 3
        a[a > 2] = 2
        a[a < 0] = 0
        b = (samples[:,i*2+1] - ylo) / (yhi - ylo) * 3
        b[b > 2] = 2
        b[b < 0] = 0
        # assign each grid a index, compute global index by shifting
        key = key * 3 + a.astype(int)
        key = key * 3 + b.astype(int)
        print(np.unique(a.astype(int)))
        print(np.unique(b.astype(int)))

    # count how many of the grid parts are filled:
    keys, discovery_indices = np.unique(key, return_index=True)
    discovery_indices.sort()
    return discovery_indices 
    #nkeys = len(np.unique(key))
    #return nkeys #, 3**(len(borders) * 2)

@mem.cache
def mcmc(logfunction, x0, nsteps, sigma_p):
    samples = np.empty((nsteps, len(x0)))
    logL0 = logfunction(x0)
    naccepts = 0
    for i in tqdm.trange(nsteps):
        x1 = np.random.normal(x0, sigma_p)
        logL1 = logfunction(x1)
        if logL1 - logL0 > np.log(np.random.uniform()):
            x0, logL0 = x1, logL1
            naccepts += 1
        samples[i,:] = x0
    return samples, naccepts

from ultranest.mlfriends import MLFriends, AffineLayer

def setup_region(startpoint):
    samples, naccepts = mcmc(flat_indicator, startpoint, 40000, 1e-5)
    us = samples[::100,:]
    Ls = loglikelihood(us)
    print('unique live points:', len(np.unique(Ls)), len(us), naccepts)
    transformLayer = AffineLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    return us, Ls, region

import ultranest.stepsampler

def fetch_samples(sampler, startpoint, nsteps):
    us, Ls, region = setup_region(startpoint)
    Nlive = len(us)
    samples = np.empty((nsteps, nparams))
    ncalls = np.zeros(nsteps, dtype=int)
    nc_cum = 0
    for i in tqdm.trange(nsteps):
        u = None
        while u is None:
            u, p, L, nc = sampler.__next__(region, Lmin, us, Ls, transform, loglikelihood)
            nc_cum += nc
        samples[i,:] = u
        ncalls[i] = nc_cum
        assert L > Lmin
        # replace a live point
        k = np.random.randint(Nlive)
        if k == 0:
            print("scale:", sampler.scale, "span:", us.max(axis=0) - us.min(axis=0))
        us[k,:] = u
        Ls[k] = L
        if i % Nlive == Nlive - 1:
            # update region:
            transformLayer = AffineLayer()
            transformLayer.optimize(us, us)
            region = MLFriends(us, transformLayer)
            region.maxradiussq, region.enlarge = region.compute_enlargement()
            region.create_ellipsoid()
        
    return samples, ncalls

step_matrix=np.arange(nparams).reshape((-1, 1))
K = 10
samplers = [
    ('mh', 100000, ultranest.stepsampler.MHSampler(nsteps=K, generate_direction=ultranest.stepsampler.generate_random_direction)),
    ('regionmh', 100000, ultranest.stepsampler.MHSampler(nsteps=K, generate_direction=ultranest.stepsampler.generate_region_random_direction)),
    ('cubeslice', 10000, ultranest.stepsampler.SliceSampler(nsteps=K, generate_direction=ultranest.stepsampler.generate_cube_oriented_direction)),
    ('regionslice', 10000, ultranest.stepsampler.SliceSampler(nsteps=K, generate_direction=ultranest.stepsampler.generate_region_oriented_direction)),
    ('regionball', 10000, ultranest.stepsampler.SliceSampler(nsteps=K, generate_direction=ultranest.stepsampler.generate_region_random_direction)),
    #('seqregionslice', 10000, ultranest.stepsampler.SliceSampler(nsteps=K,
    #    generate_direction=ultranest.stepsampler.SpeedVariableGenerator(step_matrix=step_matrix, generate_direction=ultranest.stepsampler.generate_region_oriented_direction))
    #),
    #('seqcubeslice', 10000, ultranest.stepsampler.SliceSampler(nsteps=K,
    #    generate_direction=ultranest.stepsampler.SpeedVariableGenerator(step_matrix=step_matrix, generate_direction=ultranest.stepsampler.generate_cube_oriented_direction))
    #)
]

@mem.cache
def get_samples(samplername, nsteps, nparams, seed=1):
    np.random.seed(seed)
    for samplernamei, _, sampler in samplers:
        if samplername == samplername:
            tstart = time.time()
            samples, ncalls = fetch_samples(
                startpoint=startpoint,
                sampler=sampler,
                nsteps=nsteps
            )
            return samples, ncalls, time.time() - tstart
    assert False

def main():
    print('start point:', loglikelihood(np.asarray([startpoint])), startpoint, Lmin)
    print('start point:', flat_indicator(startpoint), startpoint)
    # generate live points

    for samplername, nsteps, sampler in samplers:
        print("checking sampler: %s" % samplername, sampler)
        samples, ncalls, T = get_samples(samplername, nsteps, nparams)

        #samples, naccepts = mcmc(flat_indicator, startpoint, nsteps, 0.1)
        #print('acceptance rate: %.2f%%' % (naccepts * 100 / nsteps))

        #u = np.random.uniform(size=(1000000, nparams))
        #L = loglikelihood(u)
        #i = np.argmax(L)
        #print('guessed point:', L[i], u[i,:])

        discovery_indices = frac_filled(samples)
        print("discovery indices:", discovery_indices, "average cost per step:", ncalls[-1] / nsteps)
        plt.plot(discovery_indices * ncalls[-1] / nsteps, np.arange(len(discovery_indices)), label=samplername)
        plt.legend(loc='best')
        plt.xscale('log')
        plt.xlim(1, None)
        plt.savefig('letters_discovery.pdf', bbox_inches='tight')
        
        import corner
        corner.corner(samples, truths=startpoint)
        plt.savefig('letters_sampled_%s.pdf' % samplername, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
