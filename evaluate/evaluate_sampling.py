import numpy as np
import matplotlib.pyplot as plt
from mininest.mlfriends import ScalingLayer, AffineLayer, MLFriends
from mininest.stepsampler import RegionMHSampler, CubeMHSampler
from mininest.stepsampler import CubeSliceSampler, RegionSliceSampler, RegionBallSliceSampler, RegionSequentialSliceSampler
#from mininest.stepsampler import DESampler
from mininest.stepsampler import OtherSamplerProxy, SamplingPathSliceSampler, SamplingPathStepSampler
import tqdm
import joblib
from problems import transform, get_problem

mem = joblib.Memory('.', verbose=False)

def quantify_step(a, b):
    # euclidean step distance
    stepsize = ((a - b)**2).sum()
    # assuming a 
    center = 0.5
    da = a - center
    db = b - center
    ra = ((da**2).sum())**0.5
    rb = ((db**2).sum())**0.5
    # compute angle between vectors da, db
    angular_step = np.arccos(np.dot(da, db) / (ra * rb))
    # compute step in radial direction
    radial_step = np.abs(ra - rb)
    return [stepsize, angular_step, radial_step]

@mem.cache
def evaluate_warmed_sampler(problemname, ndim, nlive, nsteps, sampler):
    loglike, grad, volume, warmup = get_problem(problemname, ndim=ndim)
    if hasattr(sampler, 'set_gradient'):
        sampler.set_gradient(grad)
    np.random.seed(1)
    us = np.array([warmup(ndim) for i in range(nlive)])
    Ls = np.array([loglike(u) for u in us])
    vol0 = max((volume(Li, ndim) for Li in Ls))
    nwarmup = 3 * nlive
    
    if ndim > 1:
        transformLayer = AffineLayer()
    else:
        transformLayer = ScalingLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=vol0)
    
    Lsequence = []
    stepsequence = []
    ncalls = 0
    for i in tqdm.trange(nsteps + nwarmup):
        if i % int(nlive * 0.2) == 0:
            minvol = (1 - 1./nlive)**i * vol0
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
            if i > nwarmup:
                ncalls += nc
            if logl is not None:
                break
        
        if i > nwarmup:
            Lsequence.append(Lmin)
            stepsequence.append(quantify_step(us[sampler.starti,:], u))

        us[j,:] = u
        Ls[j] = logl
    
    Lsequence = np.asarray(Lsequence)
    return Lsequence, ncalls, np.array(stepsequence)

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
    
    samplers = [
        #CubeMHSampler(nsteps=16), #CubeMHSampler(nsteps=4), CubeMHSampler(nsteps=1),
        #RegionMHSampler(nsteps=16), #RegionMHSampler(nsteps=4), RegionMHSampler(nsteps=1),
        ##DESampler(nsteps=16), DESampler(nsteps=4), #DESampler(nsteps=1),
        #CubeSliceSampler(nsteps=16), CubeSliceSampler(nsteps=4), CubeSliceSampler(nsteps=1),
        RegionSliceSampler(nsteps=2*ndim), RegionSliceSampler(nsteps=ndim), RegionSliceSampler(nsteps=max(1, ndim//2)),
        #RegionBallSliceSampler(nsteps=16), RegionBallSliceSampler(nsteps=4), RegionBallSliceSampler(nsteps=1),
        #RegionBallSliceSampler(nsteps=2*ndim), RegionBallSliceSampler(nsteps=ndim), RegionBallSliceSampler(nsteps=max(1, ndim//2)),
        #RegionSequentialSliceSampler(nsteps=2*ndim), RegionSequentialSliceSampler(nsteps=ndim), RegionSequentialSliceSampler(nsteps=max(1, ndim//2)),
        
        #SamplingPathSliceSampler(nsteps=16), SamplingPathSliceSampler(nsteps=4), SamplingPathSliceSampler(nsteps=1),
        #SamplingPathStepSampler(nresets=8, nsteps=ndim * 2),
        SamplingPathStepSampler(nresets=16, nsteps=ndim * 8),
        SamplingPathStepSampler(nresets=8, nsteps=ndim * 4),
        #SamplingPathStepSampler(nresets=4, nsteps=ndim * 2),
        #SamplingPathStepSampler(nresets=2, nsteps=ndim),
        #SamplingPathStepSampler(nresets=3, nsteps=10),
        #SamplingPathStepSampler(nresets=4, nsteps=ndim * 8),
        #SamplingPathStepSampler(nresets=2, nsteps=ndim * 2, log=True),
        #OtherSamplerProxy(nnewdirections=8, nsteps=64, sampler='simple'),
        #OtherSamplerProxy(nnewdirections=4, nsteps=16, sampler='simple'),
        #OtherSamplerProxy(nnewdirections=16, nsteps=1, sampler='simple'),
        #OtherSamplerProxy(nsteps=16, sampler='bisect'),
    ]
    if ndim < 14:
        samplers.insert(0, MLFriendsSampler())
    colors = {}
    linestyles = {1:':', 4:'--', 16:'-', 32:'-', 64:'-', -1:'-'}
    markers = {1:'x', 4:'^', 16:'o', 32:'s', 64:'s', -1:'o'}
    for isteps, ls, m in (max(1, ndim // 2), ':', 'x'), (ndim, '--', '^'), (ndim * 2, '-', 'o'), (ndim * 4, '-.', '^'), (ndim * 8, '-', 'v'), (ndim * 16, '-', '>'):
        if isteps not in markers:
            markers[isteps] = m
        if isteps not in linestyles:
            linestyles[isteps] = ls
    Lsequence_ref = None
    label_ref = None
    axL = plt.figure('Lseq').gca()
    axS = plt.figure('shrinkage').gca()
    axspeed = plt.figure('speed').gca()
    plt.figure('stepsize', figsize=(14, 6))
    axstep1 = plt.subplot(1, 3, 1)
    axstep2 = plt.subplot(1, 3, 2)
    axstep3 = plt.subplot(1, 3, 3)
    for sampler in samplers:
        print("evaluating sampler: %s" % sampler)
        Lsequence, ncalls, steps = evaluate_warmed_sampler(problemname, ndim, nlive, nsteps, sampler)
        
        loglike, grad, volume, warmup = get_problem(problemname, ndim=ndim)
        assert np.isfinite(Lsequence).all(), Lsequence
        vol = np.asarray([volume(Li, ndim) for Li in Lsequence])
        assert np.isfinite(vol).any(), ("Sampler has not reached interesting likelihoods", vol, Lsequence)
        shrinkage = 1 - (vol[np.isfinite(vol)][1:] / vol[np.isfinite(vol)][:-1])**(1. / ndim)
        
        fullsamplername = str(sampler)
        samplername = fullsamplername.split('(')[0]
        
        label = fullsamplername + ' %d evals' % ncalls
        if Lsequence_ref is None:
            label_ref = label
            Lsequence_ref = Lsequence
            ls = '-'
            color = 'pink'
        else:
            color = colors.get(samplername)
            ls = linestyles[sampler.nsteps]
            l, = axL.plot(Lsequence_ref, Lsequence, label=label, color=color, linestyle=ls, lw=1)
            colors[samplername] = l.get_color()
        
        # convert to a uniformly distributed variable, according to expectations
        cdf_expected = 1 - (1 - shrinkage)**(ndim * nlive)
        axS.hist(cdf_expected, cumulative=True, density=True, 
            histtype='step', bins=np.linspace(0, 1, 4000),
            label=label, color=color, ls=ls
        )
        print("%s shrunk %.4f, from %d shrinkage samples" % (fullsamplername, cdf_expected.mean(), len(shrinkage)))
        axspeed.plot(cdf_expected.mean(), ncalls, markers[sampler.nsteps], label=label, color=color)
        
        stepsizesq, angular_step, radial_step = steps.transpose()
        assert len(stepsizesq) == len(Lsequence), (len(stepsizesq), len(Lsequence))
        # here we estimate the volume differently: from the expected shrinkage per iteration
        it = np.arange(len(stepsizesq))
        vol = (1 - 1. / nlive)**it
        assert np.isfinite(vol).all(), vol
        assert (vol > 0).all(), vol
        assert (vol <= 1).all(), vol
        relstepsize = stepsizesq**0.5 / vol**(1. / ndim)
        relradial_step = radial_step / vol**(1. / ndim)
        axstep1.hist(relstepsize[np.isfinite(relstepsize)], cumulative=True, density=True, 
            histtype='step',
            label=label, color=color, ls=ls)
        axstep2.hist(angular_step, cumulative=True, density=True, 
            histtype='step', 
            label=label, color=color, ls=ls)
        axstep3.hist(relradial_step, cumulative=True, density=True, 
            histtype='step', 
            label=label, color=color, ls=ls)
    
    print('range:', Lsequence_ref[0], Lsequence_ref[-1])
    axL.plot([Lsequence_ref[0], Lsequence_ref[-1]], [Lsequence_ref[0], Lsequence_ref[-1]], '-', color='k', lw=1, label=label_ref)
    axL.set_xlabel('logL (reference)')
    axL.set_ylabel('logL')
    lo, hi = Lsequence_ref[int(len(Lsequence_ref)*0.1)], Lsequence_ref[-1]
    axL.set_xlim(lo, hi)
    axL.set_ylim(lo, hi)
    axL.legend(loc='best', prop=dict(size=6))
    filename = 'evaluate_sampling_%s_%dd_N%d_L.pdf' % (args.problem, ndim, nlive)
    print("plotting to %s ..." % filename)
    plt.figure('Lseq')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    plt.figure('shrinkage')
    plt.xlabel('Shrinkage Volume')
    plt.ylabel('Cumulative Distribution')
    plt.xlim(0, 1)
    plt.plot([0,1], [0,1], '--', color='k')
    plt.legend(loc='best', prop=dict(size=6))
    filename = 'evaluate_sampling_%s_%dd_N%d_shrinkage.pdf' % (args.problem, ndim, nlive)
    print("plotting to %s ..." % filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    plt.figure('speed')
    plt.xlabel('Bias')
    plt.ylabel('# of function evaluations')
    plt.yscale('log')
    plt.legend(loc='best', prop=dict(size=6), fancybox=True, framealpha=0.5)
    filename = 'evaluate_sampling_%s_%dd_N%d_speed.pdf' % (args.problem, ndim, nlive)
    print("plotting to %s ..." % filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    plt.figure('stepsize')
    axstep1.set_ylabel('Cumulative Distribution')
    axstep1.set_xlabel('Euclidean distance')
    axstep1.legend(loc='lower right', prop=dict(size=6))
    axstep2.set_ylabel('Cumulative Distribution')
    axstep2.set_xlabel('Angular distance')
    #axstep2.legend(loc='best', prop=dict(size=6))
    axstep3.set_ylabel('Cumulative Distribution')
    axstep3.set_xlabel('Radial distance')
    #axstep3.legend(loc='best', prop=dict(size=6))
    filename = 'evaluate_sampling_%s_%dd_N%d_step.pdf' % (args.problem, ndim, nlive)
    print("plotting to %s ..." % filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=200)
    parser.add_argument("--problem", type=str, default="circgauss")
    parser.add_argument('--nsteps', type=int, default=1000)

    args = parser.parse_args()
    main(args)

