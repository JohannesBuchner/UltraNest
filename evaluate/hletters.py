from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import sys
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

words = {len(w)*2:w for w in r'W II NEST ULTRANEST SSSSS NESTED-SAMPLING-WITH-ULTRANEST'.split()}
print("available words:", words)
word = words[int(sys.argv[1])]
#word = 'II' # 4d
#word = 'NEST' # 8d
#word = 'ULTRANEST' # 18d
#word = 'NESTED-SAMPLING-WITH-ULTRANEST' # 60d
characters = [sample_character(c, 'comic') for c in word]

#for i, c in enumerate(characters):
#    plt.subplot(len(characters), 1, i+1)
#    plt.imshow(c.T >= c.max() / 2)
#plt.savefig('letters.png', bbox_inches='tight')
#plt.close()

startpoint = []
borders = []
interpolators = []
expect_filled = 1
Nexpect_filled = []
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
    Nexpect_filled.append(a.astype(int) * 3 + b.astype(int))
    expect_filled *= len(np.unique(Nexpect_filled[-1]))
    
    startpoint += [i[0] * xscale, j[0] * yscale]
    borders.append((ylo * yscale, yhi * yscale, xlo * xscale, xhi * xscale))
    print('maxval:', c.max(), 'border:', borders[-1], 'start guess:', startpoint[-2:])
    interpolators.append(
        scipy.interpolate.RegularGridInterpolator(
            # flip x and y, to correct for transposition
            (np.linspace(0, 1, c.shape[1]), np.linspace(0, 1, c.shape[0])), 
            # reverse y-axis, to correct for transposition
            c[:,::-1],
            'linear', bounds_error=True)
    )
    #masks.append(floodfill_distance(c[:,::-1] >= c.max() / 2), i[0], j[0])
    Lmin += -0.5 * (256 - interpolators[-1](startpoint[-2:]))**2
    print("   ", interpolators[-1]((i[0] * xscale, j[0] * yscale)), c.max() / 10. * 0.4)

Lmin = -0.5 * (256 - 256 // 3)**2
plt.figure(figsize=(2+len(characters), 4))
for i, c in enumerate(characters):
    plt.subplot(1, len(characters), i + 1)
    plt.imshow(c.T > 256 // 3)
plt.savefig('letters.png', bbox_inches='tight')
plt.close()

startpoint = np.array(startpoint)
paramnames = ['p%d' % (i+1) for i in range(len(characters)*2)]
nparams = len(paramnames)

def loglikelihood(params):
    likes = np.zeros((len(interpolators), params.shape[0]))
    for i, interpolator in enumerate(interpolators):
        likes[i] = -0.5 * (256 - interpolator(params[:, i*2:(i+1)*2]))**2
    return np.min(likes, axis=0)

def transform(x): return x

def flat_indicator(params):
    if (params > 0).all() and (params < 1).all():
        for i, interpolator in enumerate(interpolators):
            if interpolator(params[i*2:(i+1)*2]) < 256 // 3:
                return -np.inf
        return 0.
    return -np.inf

def calc_keys(samples):
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
    return key

def frac_filled(samples):
    key = calc_keys(samples)

    # count how many of the grid parts are filled:
    keys, discovery_indices = np.unique(key, return_index=True)
    discovery_indices.sort()
    return discovery_indices 

"""
def TVnorm(samples):
    key = calc_keys(samples)
    
    Ntot = len(samples)
    # count how many of the grid parts are filled:
    
    keys, discovery_indices, counts = np.unique(key, return_index=True, return_counts=True)

    KL = 0.0
    TV = 0.0
    for key, N in zip(keys, counts):
        Nexpect = 1.0
        for keyexpect in Nexpect_filled[::-1]:
            ab = key % 9
            key //= 9
            Nexpect *= (keyexpect == ab).mean()
        KL += N / Ntot * np.log10(N / Ntot - Nexpect)
        TV = max(TV, np.abs(N / Ntot - Nexpect))
    
    return KL
"""

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
    print("finding some initial live points...")
    samples, naccepts = mcmc(flat_indicator, startpoint, 40000, 1e-5)
    us = samples[::100,:]
    Ls = loglikelihood(us)
    Nuniq = len(np.unique(Ls))
    print('unique live points:', Nuniq, len(us), naccepts)
    assert Nuniq > 1
    transformLayer = AffineLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement()
    region.create_ellipsoid()
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
            u, p, L, nc = sampler.__next__(region, Lmin, us, Ls, transform, loglikelihood, ndraw=10)
            nc_cum += nc
        samples[i,:] = u
        ncalls[i] = nc_cum
        assert L > Lmin, (L, Lmin)
        # replace a live point
        k = np.random.randint(Nlive)
        if k == 0:
            print("scale:", getattr(sampler, 'scale', None), "span:", us.max(axis=0) - us.min(axis=0))
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

from ultranest.stepsampler import generate_cube_oriented_direction, generate_region_oriented_direction, generate_region_random_direction, generate_random_direction
from ultranest.stepsampler import SliceSampler, MHSampler, OrthogonalProposalGenerator, AHARMSampler
step_matrix=np.arange(nparams).reshape((-1, 1))
K = max(10, nparams)

samplers = [
    ('mh', 10000 * nparams, MHSampler(nsteps=K, generate_direction= generate_random_direction)),
    ##('regionmh', 100000,  MHSampler(nsteps=K, generate_direction= generate_region_random_direction)),
    ('cubeslice', 10000 * nparams,  SliceSampler(nsteps=K, generate_direction=generate_cube_oriented_direction)),
    #('regionslice', 1000 * nparams,  SliceSampler(nsteps=K, generate_direction=generate_region_oriented_direction)),
    ('regionball', 10000 * nparams,  SliceSampler(nsteps=K, generate_direction=generate_region_random_direction)),
    ('acubeslice', 10000 * nparams,  AHARMSampler(nsteps=K, generate_direction=generate_cube_oriented_direction, orthogonalise=False)),
    #('cubeslice-orth', 1000 * nparams,  SliceSampler(nsteps=K, generate_direction=OrthogonalProposalGenerator(generate_cube_oriented_direction))),
    #('regionslice-orth', 1000 * nparams,  SliceSampler(nsteps=K, generate_direction=OrthogonalProposalGenerator(generate_region_oriented_direction))),
    #('regionball-orth', 1000 * nparams,  SliceSampler(nsteps=K, generate_direction=OrthogonalProposalGenerator(generate_region_random_direction))),
    #('seqregionslice', 100000 * nparams,  SliceSampler(nsteps=K,
    #    generate_direction= SpeedVariableGenerator(step_matrix=step_matrix, generate_direction= generate_region_oriented_direction))
    #),
    #('seqcubeslice', 10000,  SliceSampler(nsteps=K,
    #    generate_direction= SpeedVariableGenerator(step_matrix=step_matrix, generate_direction= generate_cube_oriented_direction))
    #)
]

@mem.cache
def get_samples(samplername, nsteps, nparams, seed=1):
    np.random.seed(seed)
    for samplernamei, _, sampler in samplers:
        if samplername == samplernamei:
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
    assert flat_indicator(startpoint) == 0.0
    # generate live points

    for samplername, nsteps, sampler in samplers:
        print("checking sampler: %s" % samplername, sampler)
        l = None
        samples, ncalls, T = get_samples(samplername, nsteps, nparams)
        discovery_indices = frac_filled(samples) + 1
        ndiscovered = np.arange(len(discovery_indices)) + 1
        cost = discovery_indices * ncalls[-1] / nsteps
        print("discovery indices:", discovery_indices, "average cost per step:", ncalls[-1] / nsteps)
        if l is None:
            l, = plt.plot(cost, ndiscovered, label=samplername)
        else:
            plt.plot(cost, ndiscovered, color=l.get_color())
        print(cost.max())

        plt.legend(loc='best', title='%s (%d dim)' % (word, nparams), prop=dict(size=8))
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim(1, None)
        plt.xlabel('Number of model evaluations')
        plt.ylabel('Number of regions discovered')
        plt.savefig('hletters_discovery_%d.pdf' % nparams, bbox_inches='tight')
        
        import corner
        corner.corner(samples, truths=startpoint)
        plt.savefig('hletters_sampled_%s.pdf' % samplername, bbox_inches='tight')
        plt.close()
        
        """plt.figure()
        r, pvalue = scipy.stats.spearmanr(samples)
        r[np.arange(nparams),np.arange(nparams)] = 0
        plt.imshow(r)
        plt.colorbar()
        plt.savefig('letters_corr.pdf', bbox_inches='tight')
        plt.close()"""

if __name__ == '__main__':
    main()
