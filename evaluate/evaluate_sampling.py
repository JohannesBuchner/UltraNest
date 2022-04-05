import numpy as np
import matplotlib.pyplot as plt
from ultranest.mlfriends import ScalingLayer, AffineLayer, RobustEllipsoidRegion
from ultranest.stepsampler import RegionMHSampler, CubeMHSampler
from ultranest.stepsampler import CubeSliceSampler, RegionSliceSampler, RegionBallSliceSampler, RegionSequentialSliceSampler, SpeedVariableRegionSliceSampler
#from ultranest.stepsampler import AHARMSampler
#from ultranest.stepsampler import OtherSamplerProxy, SamplingPathSliceSampler, SamplingPathStepSampler
#from ultranest.stepsampler import GeodesicSliceSampler, RegionGeodesicSliceSampler
import tqdm
import joblib
import warnings, traceback
from problems import transform, get_problem

mem = joblib.Memory('.', verbose=False)

def quantify_step(a, b):
    # euclidean step distance
    stepsize = np.linalg.norm(a - b)
    # assuming a 
    center = 0.5
    da = a - center
    db = b - center
    ra = np.linalg.norm(da)
    rb = np.linalg.norm(db)
    # compute angle between vectors da, db
    angular_step = np.arccos(np.dot(da, db) / (ra * rb))
    # compute step in radial direction
    radial_step = np.abs(ra - rb)
    return [stepsize, angular_step, radial_step]

@mem.cache
def evaluate_warmed_sampler(problemname, ndim, nlive, nsteps, sampler, seed=1, region_class=RobustEllipsoidRegion):
    loglike, grad, volume, warmup = get_problem(problemname, ndim=ndim)
    if hasattr(sampler, 'set_gradient'):
        sampler.set_gradient(grad)
    np.random.seed(seed)
    def multi_loglike(xs):
        return np.asarray([loglike(x) for x in xs])
    us = np.array([warmup(ndim) for i in range(nlive)])
    Ls = np.array([loglike(u) for u in us])
    vol0 = volume(Ls.min(), ndim)
    nwarmup = 3 * nlive
    
    if ndim > 1:
        transformLayer = AffineLayer()
    else:
        transformLayer = ScalingLayer()
    transformLayer.optimize(us, us)
    region = region_class(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=vol0)
    assert region.ellipsoid_center is not None
    sampler.region_changed(Ls, region)
    
    Lsequence = []
    stepsequence = []
    ncalls = 0
    for i in tqdm.trange(nsteps + nwarmup):
        if i % int(nlive * 0.2) == 0:
            minvol = (1 - 1./nlive)**i * vol0
            with warnings.catch_warnings(), np.errstate(all='raise'):
                try:
                    nextTransformLayer = transformLayer.create_new(us, region.maxradiussq, minvol=minvol)
                    nextregion = region_class(us, nextTransformLayer)
                    nextregion.maxradiussq, nextregion.enlarge = nextregion.compute_enlargement(nbootstraps=30)
                    if isinstance(nextregion, RobustEllipsoidRegion) or nextregion.estimate_volume() <= region.estimate_volume():
                        nextregion.create_ellipsoid(minvol=minvol)
                        region = nextregion
                        transformLayer = region.transformLayer
                        assert region.ellipsoid_center is not None
                        sampler.region_changed(Ls, region)
                except Warning as w:
                    print("not updating region because: %s" % w)
                except FloatingPointError as e:
                    print("not updating region because: %s" % e)
                except np.linalg.LinAlgError as e:
                    print("not updating region because: %s" % e)
        
        # replace lowest likelihood point
        j = np.argmin(Ls)
        Lmin = float(Ls[j])
        while True:
            u, v, logl, nc = sampler.__next__(region, Lmin, us, Ls, transform, multi_loglike)
            if i > nwarmup:
                ncalls += nc
            if logl is not None:
                assert np.isfinite(u).all(), u
                assert np.isfinite(v).all(), v
                assert np.isfinite(logl), logl
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
        self.adaptive_nsteps = False
    
    def __next__(self, region, Lmin, us, Ls, transform, loglike):
        u = region.sample(nsamples=self.ndraw)
        nu = u.shape[0]
        self.starti = np.random.randint(len(us))
        if nu > 0:
            u = u[:1,:]
            v = transform(u)
            logl = loglike(v)[0]
            accepted = logl > Lmin
            if accepted:
                return u[0], v[0], logl, 1
            return None, None, None, 1
        return None, None, None, 0
        
    def __str__(self):
        return 'MLFriends'
    def plot(self, filename):
        pass
    def region_changed(self, Ls, region):
        pass

def main(args):
    nlive = args.num_live_points
    ndim = args.x_dim
    nsteps = args.nsteps
    problemname = args.problem
    
    samplers = [
        #CubeMHSampler(nsteps=16), #CubeMHSampler(nsteps=4), CubeMHSampler(nsteps=1),
        #RegionMHSampler(nsteps=16), #RegionMHSampler(nsteps=4), RegionMHSampler(nsteps=1),
        ##DESampler(nsteps=16), DESampler(nsteps=4), #DESampler(nsteps=1),
        CubeSliceSampler(nsteps=2*ndim), #CubeSliceSampler(nsteps=ndim), CubeSliceSampler(nsteps=max(1, ndim//2)),
        #RegionSliceSampler(nsteps=ndim), RegionSliceSampler(nsteps=max(1, ndim//2)),
        #RegionSliceSampler(nsteps=2), RegionSliceSampler(nsteps=4), 
        #RegionSliceSampler(nsteps=ndim), RegionSliceSampler(nsteps=4*ndim),
        #RegionBallSliceSampler(nsteps=2*ndim), RegionBallSliceSampler(nsteps=ndim), RegionBallSliceSampler(nsteps=max(1, ndim//2)),
       # RegionSequentialSliceSampler(nsteps=2*ndim), RegionSequentialSliceSampler(nsteps=ndim), RegionSequentialSliceSampler(nsteps=max(1, ndim//2)),

        #SpeedVariableRegionSliceSampler([Ellipsis]*ndim), SpeedVariableRegionSliceSampler([slice(i, ndim) for i in range(ndim)]),
        #SpeedVariableRegionSliceSampler([Ellipsis]*ndim + [slice(1 + ndim//2, None)]*ndim), 
    ]
    if ndim < 14:
        samplers.insert(0, MLFriendsSampler())
    colors = {}
    linestyles = {1:':', 2:':', 4:'--', 16:'-', 32:'-', 64:'-', -1:'-'}
    markers = {1:'x', 2:'x', 4:'^', 16:'o', 32:'s', 64:'s', -1:'o'}
    for isteps, ls, m in (max(1, ndim // 2), ':', 'x'), (ndim, '--', '^'), (ndim+1, '--', '^'), (ndim * 2, '-', 'o'), (ndim * 4, '-.', '^'), (ndim * 8, '-', 'v'), (ndim * 16, '-', '>'):
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
    lastspeed = None, None, None
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
            ls = '-' if sampler.adaptive_nsteps else linestyles[sampler.nsteps]
            l, = axL.plot(Lsequence_ref, Lsequence, label=label, color=color, linestyle=ls, lw=1)
            colors[samplername] = l.get_color()
        
        # convert to a uniformly distributed variable, according to expectations
        cdf_expected = 1 - (1 - shrinkage)**(ndim * nlive)
        axS.hist(cdf_expected, cumulative=True, density=True, 
            histtype='step', bins=np.linspace(0, 1, 4000),
            label=label, color=color, ls=ls
        )
        print("%s shrunk %.4f, from %d shrinkage samples" % (fullsamplername, cdf_expected.mean(), len(shrinkage)))
        axspeed.plot(cdf_expected.mean(), ncalls, markers[1 if sampler.adaptive_nsteps else sampler.nsteps], label=label, color=color)
        if lastspeed[0] == samplername:
            axspeed.plot([lastspeed[1], cdf_expected.mean()], [lastspeed[2], ncalls], '-', color=color)
        lastspeed = [samplername, cdf_expected.mean(), ncalls]
        
        stepsize, angular_step, radial_step = steps.transpose()
        assert len(stepsize) == len(Lsequence), (len(stepsize), len(Lsequence))
        # here we estimate the volume differently: from the expected shrinkage per iteration
        it = np.arange(len(stepsizesq))
        vol = (1 - 1. / nlive)**it
        assert np.isfinite(vol).all(), vol
        assert (vol > 0).all(), vol
        assert (vol <= 1).all(), vol
        relstepsize = stepsize / vol**(1. / ndim)
        relradial_step = radial_step / vol**(1. / ndim)
        axstep1.hist(relstepsize[np.isfinite(relstepsize)], bins=1000, cumulative=True, density=True, 
            histtype='step',
            label=label, color=color, ls=ls)
        axstep2.hist(angular_step, bins=1000, cumulative=True, density=True, 
            histtype='step', 
            label=label, color=color, ls=ls)
        axstep3.hist(relradial_step, bins=1000, cumulative=True, density=True, 
            histtype='step', 
            label=label, color=color, ls=ls)
        sampler.plot(filename = 'evaluate_sampling_%s_%dd_N%d_%s.png' % (args.problem, ndim, nlive, samplername))
    
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
    plt.legend(loc='upper left', prop=dict(size=6))
    filename = 'evaluate_sampling_%s_%dd_N%d_shrinkage.pdf' % (args.problem, ndim, nlive)
    print("plotting to %s ..." % filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    plt.figure('speed')
    plt.xlabel('Bias')
    plt.ylabel('# of function evaluations')
    plt.yscale('log')
    lo, hi = plt.xlim()
    hi = max(0.5 - lo, hi - 0.5, 0.04)
    plt.xlim(0.5 - hi, 0.5 + hi)
    lo, hi = plt.ylim()
    plt.vlines(0.5, lo, hi)
    plt.ylim(lo, hi)
    plt.legend(loc='best', prop=dict(size=6), fancybox=True, framealpha=0.5)
    filename = 'evaluate_sampling_%s_%dd_N%d_speed.pdf' % (args.problem, ndim, nlive)
    print("plotting to %s ..." % filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), bbox_inches='tight')
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
