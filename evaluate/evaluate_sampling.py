import warnings
import time
import numpy as np
import tqdm
import joblib
import matplotlib.pyplot as plt
import scipy.stats

from ultranest.mlfriends import ScalingLayer, AffineLayer, MLFriends, RobustEllipsoidRegion
from ultranest.stepsampler import RegionMHSampler, CubeMHSampler, DEMCSampler
from ultranest.stepsampler import CubeSliceSampler, RegionSliceSampler, RegionBallSliceSampler, RegionSequentialSliceSampler, SpeedVariableRegionSliceSampler
from ultranest.stepsampler import AHARMSampler, generate_region_random_direction, generate_random_direction, generate_region_oriented_direction
from ultranest.ordertest import UniformOrderAccumulator

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

def init_region(ndim, us, vol0):
    if ndim > 1:
        transformLayer = AffineLayer()
    else:
        transformLayer = ScalingLayer()
    transformLayer.optimize(us, us)
    if ndim > 30:
        region = RobustEllipsoidRegion(us, transformLayer)
        region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    else:
        region = MLFriends(us, transformLayer)
        region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=vol0)
    assert region.ellipsoid_center is not None
    return region

def update_region(ndim, region, us, minvol):
    with warnings.catch_warnings(), np.errstate(all='raise'):
        try:
            if ndim > 30:
                nextTransformLayer = region.transformLayer.create_new(us, np.inf, minvol=minvol)
                nextregion = RobustEllipsoidRegion(us, nextTransformLayer)
                nextregion.maxradiussq, nextregion.enlarge = region.compute_enlargement(nbootstraps=30)
                nextregion.create_ellipsoid(minvol=minvol)
            else:
                nextTransformLayer = region.transformLayer.create_new(us, region.maxradiussq, minvol=minvol)
                nextregion = MLFriends(us, nextTransformLayer)
                nextregion.maxradiussq, nextregion.enlarge = region.compute_enlargement(nbootstraps=30)
                nextregion.create_ellipsoid(minvol=minvol)
            if nextregion.estimate_volume() <= region.estimate_volume():
                assert region.ellipsoid_center is not None
                # print("region updated", nextregion.estimate_volume(), nextregion.enlarge)
                return nextregion, True
        except Warning as w:
            print("not updating region because: %s" % w)
        except FloatingPointError as e:
            print("not updating region because: %s" % e)
        except np.linalg.LinAlgError as e:
            print("not updating region because: %s" % e)
    return region, False

@mem.cache(ignore=['sampler'])
def evaluate_warmed_sampler(problemname, ndim, nlive, nsteps, samplername, sampler, seed=1):
    loglike, grad, volume, warmup = get_problem(problemname, ndim=ndim)
    if hasattr(sampler, 'set_gradient'):
        sampler.set_gradient(grad)
    np.random.seed(seed)
    def multi_loglike(xs):
        return np.asarray([loglike(x) for x in xs])
    us = np.array([warmup(ndim) for i in range(nlive)])
    Ls = np.array([loglike(u) for u in us])
    vol0 = max((volume(Li, ndim) for Li in Ls))
    nwarmup = 3 * nlive
    
    region = init_region(ndim, us, vol0)
    sampler.region_changed(Ls, region)
    
    Lsequence = []
    stepsequence = []
    ranks = []
    t0 = time.time()
    update_fraction = 0.2
    if ndim > 40:
        update_fraction = 0.5
    ncalls = 0
    for i in tqdm.trange(nsteps + nwarmup):
        if i % int(nlive * update_fraction) == 0:
            t1 = time.time()
            minvol = (1 - 1./nlive)**i * vol0
            t0 += time.time() - t1
            region, region_updated = update_region(ndim, region, us, minvol)
            if region_updated:
                sampler.region_changed(Ls, region)
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
            ranks.append((Ls < logl).sum())

        us[j,:] = u
        Ls[j] = logl
    
    Lsequence = np.asarray(Lsequence)
    return Lsequence, ncalls, np.array(stepsequence), np.array(ranks), time.time() - t0

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
    num_eval_steps = args.nsteps
    num_total_steps = 20000
    problemname = args.problem
    
    samplerclasses = [CubeMHSampler, CubeSliceSampler, RegionSliceSampler, RegionBallSliceSampler]
    nsteps_set = sorted({1,2,4,16,max(16,ndim//2),max(16,ndim),max(16,ndim*2)}, reverse=True)
    #nsteps_set = sorted({1,2,4}, reverse=True)
    samplers = [
        (samplerclass.__name__.lower().replace('sampler','') + '-%d' % nsteps, samplerclass(nsteps=nsteps)) 
        for samplerclass in samplerclasses for nsteps in nsteps_set]
    #for name, generate_direction in ('regionball', generate_region_random_direction), ('ball', generate_random_direction):
    #    #, generate_region_oriented_direction, generate_cube_oriented_direction:
    #    samplers += [('a_%s-%d' % (name, nsteps), AHARMSampler(generate_direction=generate_direction, nsteps=nsteps)) for nsteps in nsteps_set]
    for name, generate_direction in ('regionball', generate_region_random_direction), ('regionslice', generate_region_oriented_direction),:
        samplers += [('a_%sE16-%d' % (name, nsteps), AHARMSampler(generate_direction=generate_direction, nsteps=nsteps, enlargement_factor=16)) for nsteps in nsteps_set]
    for name, generate_direction in ('regionslice', generate_region_oriented_direction),:
        samplers += [('a_%sE4-%d' % (name, nsteps), AHARMSampler(generate_direction=generate_direction, nsteps=nsteps, enlargement_factor=4)) for nsteps in nsteps_set]
    #samplers += [
    #    (samplerclass.__name__.lower().replace('sampler','') + '-adaptMD', samplerclass(nsteps=400, adaptive_nsteps='move-distance')) 
    #    for samplerclass in samplerclasses]
    if ndim < 14:
        samplers.insert(0, ('MLFriends', MLFriendsSampler()))
    colors = {}
    linestyles = {1:':', 2:':', 4:'--', 16:'-', 32:'-', 64:'-', -1:'-'}
    markers = {1:'x', 2:'x', 4:'^', 16:'o', 32:'s', 64:'s', -1:'s'}
    for isteps, ls, m in (max(1, ndim // 2), ':', 'x'), (ndim, '--', '^'), (ndim+1, '--', '^'), (ndim * 2, '-', 'o'), (ndim * 4, '-.', '^'), (ndim * 8, '-', 'v'), (ndim * 16, '-', '>'):
        if isteps not in markers:
            markers[isteps] = m
        if isteps not in linestyles:
            linestyles[isteps] = ls
    Lsequence_ref = None
    label_ref = None
    axL = plt.figure('Lseq').gca()
    axS = plt.figure('shrinkage').gca()
    axstepspeed = plt.figure('stepspeed').gca()
    axspeed = plt.figure('speed').gca()
    plt.figure('stepsize', figsize=(14, 6))
    axstep1 = plt.subplot(1, 3, 1)
    axstep2 = plt.subplot(1, 3, 2)
    axstep3 = plt.subplot(1, 3, 3)
    lastspeed = None, None, None, None
    for samplername, sampler in samplers:
        samplergroup = samplername.split('-')[0]
        if lastspeed[0] is not None and lastspeed[0] == samplergroup and lastspeed[1] < 0.005:
            print("skipping %s ..." % samplername)
            continue
        else:
            print("last:", lastspeed)
        loglike, grad, volume, warmup = get_problem(problemname, ndim=ndim)
        shrinkages = []
        ranker = UniformOrderAccumulator()
        duration = 0
        while True:
            print("evaluating sampler: %s" % samplername, "seed=%d" % (len(shrinkages)+1), '%d/%d' % (len(shrinkages) * num_eval_steps, num_total_steps))
            Lsequence, ncalls, steps, ranks_here, dt = evaluate_warmed_sampler(problemname, ndim, nlive, num_eval_steps, samplername=samplername, sampler=sampler, seed=len(shrinkages)+1)
            duration += dt
            for rank in ranks_here:
                ranker.add(rank, nlive)
            
            assert np.isfinite(Lsequence).all(), Lsequence
            vol_here = np.asarray([volume(Li, ndim) for Li in Lsequence])
            assert np.isfinite(vol_here).any(), ("Sampler has not reached interesting likelihoods", vol_here, Lsequence)
            vol_here = vol_here[np.isfinite(vol_here)]
            shrinkage_here = 1 - (vol_here[1:] / vol_here[:-1])**(1. / ndim)
            shrinkages.append(shrinkage_here)
            print('  got: %d/%d' % (len(shrinkages) * len(shrinkage_here), num_total_steps), "rank", ranker.zscore)
            if len(shrinkages) * len(shrinkage_here) >= num_total_steps - 10:
                break
        shrinkage = np.concatenate(tuple(shrinkages))
        efficiency = 100 * len(shrinkages) / ncalls
        
        label = samplername + ' %d evals' % ncalls
        if Lsequence_ref is None:
            label_ref = label
            Lsequence_ref = Lsequence
            ls = '-'
            colors[samplergroup] = color = 'pink'
        else:
            color = colors.get(samplergroup)
            ls = '-' if sampler.adaptive_nsteps else linestyles[sampler.nsteps]
            l, = axL.plot(Lsequence_ref, Lsequence, label=label, color=color, linestyle=ls, lw=1)
            colors[samplergroup] = color = l.get_color()
            #print(samplergroup, colors[samplergroup])

        # convert to a uniformly distributed variable, according to expectations
        cdf_expected = 1 - (1 - shrinkage)**(ndim * nlive)
        axS.hist(cdf_expected, cumulative=True, density=True, 
            histtype='step', bins=np.linspace(0, 1, 4000),
            label=label, color=color, ls=ls
        )
        #bias = cdf_expected.mean()
        _, bias = scipy.stats.kstest(cdf_expected, 'uniform')
        print("%s p-value:%.4f, from %d shrinkage samples" % (samplername, bias, len(shrinkage)))
        axspeed.plot(bias, ncalls, markers[-1 if sampler.adaptive_nsteps else sampler.nsteps], label=label, color=color)
        if lastspeed[0] == samplergroup:
            # connect to previous
            axspeed.plot([lastspeed[1], bias], [lastspeed[2], ncalls], '-', color=color)
        
        stepsizesq, angular_step, radial_step = steps.transpose()
        assert len(stepsizesq) == len(Lsequence), (len(stepsizesq), len(Lsequence))
        # here we estimate the volume differently: from the expected shrinkage per iteration
        it = np.arange(len(stepsizesq))
        vol = (1 - 1. / nlive)**it
        assert np.isfinite(vol).all(), vol
        assert (vol > 0).all(), vol
        assert (vol <= 1).all(), vol
        relstepsize = stepsizesq**0.5 / vol**(1. / ndim)
        avg_relstepsize = np.median(relstepsize)
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
        sampler.plot(filename = 'evaluate_sampling_%s_%dd_N%d_%s.png' % (args.problem, ndim, nlive, samplergroup))
        if bias < 0.01:
            pass
            #axstepspeed.plot(efficiency, avg_relstepsize, 'o', ms=10, mec='r', mfc='none', mew=2)
        else:
            axstepspeed.plot(
                efficiency, avg_relstepsize,
                markers[-1 if sampler.adaptive_nsteps else sampler.nsteps],
                label=label, color=color)
            if lastspeed[0] is not None and lastspeed[0] == samplergroup:
                axstepspeed.plot(
                    [efficiency, 100 * len(shrinkages) / lastspeed[2]], 
                    [avg_relstepsize, lastspeed[3]], '-', color=color)
        lastspeed = [samplergroup, bias, ncalls, np.median(relstepsize)]

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
    plt.xlabel('p-value')
    plt.ylabel('# of function evaluations')
    plt.yscale('log')
    #lo, hi = plt.xlim()
    #hi = max(0.5 - lo, hi - 0.5, 0.04)
    #plt.xlim(0.5 - hi, 0.5 + hi)
    plt.xlim(0, 1)
    lo, hi = plt.ylim()
    #plt.vlines(0.05, lo, hi)
    plt.fill_between([0, 0.05], [lo, lo], [hi, hi], alpha=0.1, color='red', lw=0)
    plt.ylim(lo, hi)
    plt.legend(loc='best', prop=dict(size=6), fancybox=True, framealpha=0.5)
    filename = 'evaluate_sampling_%s_%dd_N%d_speed.pdf' % (args.problem, ndim, nlive)
    print("plotting to %s ..." % filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()

    plt.figure('stepspeed')
    plt.ylabel('distance')
    plt.xlabel('efficiency')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', prop=dict(size=6), fancybox=True, framealpha=0.5)
    filename = 'evaluate_sampling_%s_%dd_N%d_stepspeed.pdf' % (args.problem, ndim, nlive)
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
    parser.add_argument("--num_live_points", type=int, default=400)
    parser.add_argument("--problem", type=str, default="circgauss")
    parser.add_argument('--nsteps', type=int, default=1000)

    args = parser.parse_args()
    main(args)
