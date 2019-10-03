import numpy as np
import matplotlib.pyplot as plt
from ultranest.mlfriends import ScalingLayer, AffineLayer, MLFriends
from ultranest.stepsampler import RegionMHSampler, CubeMHSampler
from ultranest.stepsampler import CubeSliceSampler, RegionSliceSampler, SamplingPathSliceSampler, SamplingPathStepSampler, OtherSamplerProxy
from ultranest.stepsampler import GeodesicSliceSampler, RegionGeodesicSliceSampler
#from ultranest.stepsampler import DESampler
import tqdm
from problems import transform, get_problem

def prepare_problem(problemname, ndim, nlive, sampler):
    loglike, grad, volume, warmup = get_problem(problemname, ndim=ndim)
    if hasattr(sampler, 'set_gradient'):
        sampler.set_gradient(grad)
    np.random.seed(1)
    us = np.random.uniform(size=(nlive, ndim))
    
    if ndim > 1:
        transformLayer = AffineLayer()
    else:
        transformLayer = ScalingLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=1.0)
    
    Ls = np.array([loglike(u) for u in us])
    ncalls = 0
    nok = 0
    i = 0
    while True:
        if i % int(nlive * 0.2) == 0:
            minvol = (1 - 1./nlive)**i
            nextTransformLayer = transformLayer.create_new(us, region.maxradiussq, minvol=minvol)
            nextregion = MLFriends(us, nextTransformLayer)
            nextregion.maxradiussq, nextregion.enlarge = nextregion.compute_enlargement(nbootstraps=30)
            if nextregion.estimate_volume() <= region.estimate_volume():
                region = nextregion
                transformLayer = region.transformLayer
            region.create_ellipsoid(minvol=minvol)
        
        # replace lowest likelihood point
        j = np.argmin(Ls)
        Lmin = float(Ls[j])
        while True:
            u, v, logl, nc = sampler.__next__(region, Lmin, us, Ls, transform, loglike)
            ncalls += nc
            if logl is not None:
                break

        us[j,:] = u
        region.u[j,:] = u
        region.unormed[j,:] = region.transformLayer.transform(u)
        Ls[j] = logl
        i = i + 1
        #print(i, Lmin, volume(Lmin, ndim))
        if np.isfinite(volume(Lmin, ndim)):
            nok += 1
        
        if nok > 2 * nlive + 1000:
            break
    return region, i, Lmin, us, Ls, transform, loglike
    

class MLFriendsSampler(object):
    def __init__(self):
        self.ndraw = 40
        self.nsteps = -1
    
    def __next__(self, region, Lmin, us, Ls, transform, loglike):
        u, father = region.sample(nsamples=self.ndraw)
        nu = u.shape[0]
        self.starti = np.random.randint(len(us))
        if nu > 0:
            u = u[0,:]
            v = transform(u)
            logl = loglike(v)
            accepted = logl > Lmin
            if accepted:
                return u, v, logl, 1
            return None, None, None, 1
        return None, None, None, 0
        
    def __str__(self):
        return 'MLFriends'

def main(args):
    nlive = args.num_live_points
    ndim = args.x_dim
    nsteps = args.nsteps
    problemname = args.problem
    #num_warmup_steps = nlive * 10
    
    np.random.seed(1)
    #sampler = MLFriendsSampler()
    #region, it, Lmin, us, Ls, transform, loglike = prepare_problem(problemname, ndim, nlive, sampler)
    
    samplers = [
        #('cubemh', CubeMHSampler(nsteps=1)),
        #('regionmh', RegionMHSampler(nsteps=1)),
        #('cubeslice', CubeSliceSampler(nsteps=1)),
        #('regionslice', RegionSliceSampler(nsteps=1)),
        #('pathslice', SamplingPathSliceSampler(nsteps=1)),
        #('pathstep', SamplingPathStepSampler(nsteps=12, nresets=12, log=True)),
        #('stepsampler', OtherSamplerProxy(nsteps=10, sampler='steps')),
        ('geodesic', GeodesicSliceSampler(nsteps=2)),
        ('regiongeodesic', RegionGeodesicSliceSampler(nsteps=2)),
    ]
    if args.sampler != 'all':
        samplers = [(name, sampler) for name, sampler in samplers if name == args.sampler]
    for samplername, sampler in samplers:
        print("exploring with %s ..." % sampler)
        region, it, Lmin, us, Ls, transform, loglike = prepare_problem(problemname, ndim, nlive, sampler)
        
        nc = 0
        starti = 0
        startu = us[starti,:]
        # take 20 steps
        print("taking %d steps..." % nsteps)
        sampler.reset()
        sampler.path = None
        for i in range(nsteps):
            ax = plt.figure(figsize=(10,10)).gca()
            filename = 'viz_%s_sampler_%s_step%02d.png' % (problemname, samplername, i+1)
            # replace lowest likelihood point
            plt.plot(us[:,0], us[:,1], 'x', ms=2, color='k')
            Lmin = Ls.min()
            #sampler.__next__(region, Lmin, us, Ls, transform, loglike, plot=True)
            plt.plot(startu[0], startu[1], 'x', ms=6, color='k')
            
            while True:

                unew = sampler.move(startu, region, plot=ax)
                mask = np.logical_and(unew > 0, unew < 1).all(axis=1)
                unew = unew[mask,:]
                mask = region.inside(unew)
                if not mask.all():
                    plt.plot(unew[~mask,0], unew[~mask,1], 'v', color='r')
                if mask.any():
                    plt.plot(unew[mask,0], unew[mask,1], '^', color='b')
                    
                    # choose first
                    j = np.where(mask)[0][0]
                    unew = unew[j,:]
                    pnew = transform(unew)
                    Lnew = loglike(pnew)
                    nc += 1
                    
                    if Lnew >= Lmin:
                        plt.plot(unew[0], unew[1], 'o ', ms=4, color='g')
                        plt.plot([startu[0], unew[0]], [startu[1], unew[1]], '--', color='green')
                        sampler.adjust_accept(True, unew, pnew, Lnew, nc)
                        startu = unew
                        break
                    else:
                        plt.plot(unew[0], unew[1], 'o', ms=4, color='orange')
                        sampler.adjust_accept(False, unew, pnew, Lnew, nc)
                else:
                    sampler.adjust_outside_region()
            
            xlo, xhi = plt.xlim()
            ylo, yhi = plt.ylim()
            lo = min(xlo, ylo)
            hi = max(xhi, yhi)
            lo, hi = 0, 1
            lo, hi = us.min(), us.max()
            lo, hi = lo - (hi - lo), hi + (hi - lo)
            plt.xlim(lo, hi)
            plt.ylim(lo, hi)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=40)
    parser.add_argument("--problem",
        choices=['circgauss', 'asymgauss', 'pyramid', 'multigauss', 'shell'])
    parser.add_argument('--nsteps', type=int, default=20)
    parser.add_argument('--sampler', default='all',
        choices=['all', 'cubemh', 'regionmh', 'cubeslice', 'regionslice'])

    args = parser.parse_args()
    main(args)

