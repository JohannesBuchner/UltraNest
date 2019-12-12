import numpy as np
#from numpy.linalg import norm
import matplotlib.pyplot as plt
#import scipy.special, scipy.stats

def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """ Compute the stop condition in the main loop
    dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

    INPUTS
    ------
    thetaminus, thetaplus: ndarray[float, ndim=1]
        under and above position
    rminus, rplus: ndarray[float, ndim=1]
        under and above momentum

    OUTPUTS
    -------
    criterion: bool
        return if the condition is valid
    """
    dtheta = thetaplus - thetaminus
    #print("stop?", dtheta, rminus, rplus, np.dot(dtheta, rminus.T), np.dot(dtheta, rplus.T))
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def step_or_reflect(theta, v, epsilon, transform, loglike, gradient, Lmin):
    # step in position:
    thetaprime = theta + epsilon * v
    # check if still inside
    mask = np.logical_and(thetaprime > 0, thetaprime < 1)
    if mask.all():
        p = transform(thetaprime.reshape((1, -1)))
        logL = loglike(p)[0]
        if logL > Lmin:
            return thetaprime, v, p[0], logL, False
    
        # outside, need to reflect
        normal = gradient(thetaprime)
        #print("reflecting with gradient", normal)
    else:
        # make a unit vector pointing inwards
        normal = np.where(thetaprime <= 0, 1, np.where(thetaprime >= 1, -1, 0))
        #print("reflecting to inside", mask, normal)
    
    # project outside gradient onto our current velocity
    # subtract that part
    vnew = v - 2 * np.dot(normal, v) * normal
    
    # if the reflection is a reverse, it cannot be helpful. Stop.
    if np.dot(v, vnew) <= 0:
        return thetaprime, vnew, None, -np.inf, True
    
    # get new location, to check if we are back in the constraint
    thetaprime2 = thetaprime + epsilon * vnew
    
    # check if inside
    mask2 = np.logical_and(thetaprime2 > 0, thetaprime2 < 1)
    if mask2.all():
        p2 = transform(thetaprime2.reshape((1, -1)))
        logL2 = loglike(p2)[0]
        #if logL2 < Lmin:
        #    #print("new point is also outside", (theta, thetaprime, thetaprime2), (v, vnew), (Lmin, logL2))
        #else:
        #    #print("recovered to inside", (theta, thetaprime, thetaprime2), (v, vnew), (Lmin, logL2))
        
        # caller needs to figure out if this is ok
        return thetaprime2, vnew, p2[0], logL2, True
    else:
        #print("new point is also outside cube", (theta, thetaprime, thetaprime2), (v, vnew))
        return thetaprime2, vnew, None, -np.inf, True


def build_tree(theta, v, direction, j, epsilon, transform, loglike, gradient, Lmin):
    """The main recursion."""
    if j == 0:
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, vprime, pprime, logpprime, reflected = step_or_reflect(
            theta=theta, v=v * direction, epsilon=epsilon, 
            transform=transform, loglike=loglike, gradient=gradient, Lmin=Lmin)
        
        #if not sprime:
        #    print("stopped trajectory:", direction, logpprime, Lmin, (theta, thetaprime, epsilon))
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime
        thetaplus = thetaprime
        
        if reflected and np.dot(v, vprime) <= 0:
            # if reversing locally, store that in can_continue, not s
            sprime = True
            #print("  reversed locally")
            can_continue = False
            vminus = v * direction
            vplus = v * direction
            v = v * direction
        else:
            # Is the point acceptable?
            sprime = logpprime > Lmin
            #if sprime:
            #    print("  -->")
            #else:
            #    print("  stuck")
            can_continue = True
            vminus = vprime * direction
            vplus = vprime * direction
            v = vprime * direction
        
        pminus = pprime
        pplus = pprime
        #print(direction, (theta, thetaprime), (v, vprime))
        
        # probability is zero if it is an invalid state
        alphaprime = 1.0 * (sprime and can_continue)
        nalphaprime = 1
        nreflectprime = reflected * 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, vminus, pminus, thetaplus, vplus, pplus, thetaprime, vprime, pprime, logpprime, sprime, can_continue, alphaprime, nalphaprime, nreflectprime = \
            build_tree(theta, v, direction, j - 1, epsilon, transform, loglike, gradient, Lmin)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if can_continue and sprime:
            if direction == -1:
                thetaminus, vminus, pminus, _, _, _, thetaprime2, vprime2, pprime2, logpprime2, sprime2, can_continue2, alphaprime2, nalphaprime2, nreflectprime2 = \
                    build_tree(thetaminus, vminus, direction, j - 1, epsilon, transform, loglike, gradient, Lmin)
            else:
                _, _, _, thetaplus, vplus, pplus, thetaprime2, vprime2, pprime2, logpprime2, sprime2, can_continue2, alphaprime2, nalphaprime2, nreflectprime2 = \
                    build_tree(thetaplus, vplus, direction, j - 1, epsilon, transform, loglike, gradient, Lmin)

            # Choose which subtree to propagate a sample up from.
            if np.random.uniform() < alphaprime2 / (alphaprime + alphaprime2):
                thetaprime = thetaprime2[:]
                vprime = vprime2[:]
                pprime = pprime2[:]
                logpprime = logpprime2
            
            # Update the stopping criterion.
            sturn = stop_criterion(thetaminus, thetaplus, vminus, vplus)
            #print(sprime, sprime2, sturn)
            #if not (sprime and sprime2 and sturn):
            #    print("sprime stop:", sprime, sprime2, sturn)
            sprime = sprime and sprime2 and sturn
            can_continue = can_continue and can_continue2
            # Update the acceptance probability statistics.
            alphaprime += alphaprime2
            nalphaprime += nalphaprime2
            nreflectprime += nreflectprime2

    return thetaminus, vminus, pminus, thetaplus, vplus, pplus, thetaprime, vprime, pprime, logpprime, sprime, can_continue, alphaprime, nalphaprime, nreflectprime

def tree_sample(theta, p, logL, v, epsilon, transform, loglike, gradient, Lmin, maxheight=np.inf):
    # initialize the tree
    thetaminus = theta
    thetaplus = theta
    vminus = v[:]
    vplus = v[:]
    alpha = 1
    nalpha = 1
    nreflect = 0
    logp = logL
    fwd_possible = True
    rwd_possible = True

    j = 0  # initial heigth j = 0
    s = True  # Main loop: will keep going until s == 0.

    while s and j < maxheight:
        # Choose a direction. -1 = backwards, 1 = forwards.
        if fwd_possible and rwd_possible:
            direction = int(2 * (np.random.uniform() < 0.5) - 1)
        elif fwd_possible:
            direction = 1
        elif rwd_possible:
            direction = -1
        else:
            print("stuck in both ends")
            break

        # Double the size of the tree.
        if direction == -1:
            thetaminus, vminus, pminus, _, _, _, thetaprime, vprime, pprime, logpprime, sprime, can_continue, alphaprime, nalphaprime, nreflectprime = \
                build_tree(thetaminus, vminus, direction, j, epsilon, transform, loglike, gradient, Lmin)
        else:
            _, _, _, thetaplus, vplus, pplus, thetaprime, vprime, pprime, logpprime, sprime, can_continue, alphaprime, nalphaprime, nreflectprime = \
                build_tree(thetaplus, vplus, direction, j, epsilon, transform, loglike, gradient, Lmin)
        
        # Use Bernoulli trial to decide whether or not to move to a
        # point from the half-tree we just generated.
        if sprime and np.random.uniform() < alphaprime / (alpha + alphaprime):
            theta = thetaprime
            p = pprime
            logp = logpprime
            v = vprime
        
        alpha += alphaprime
        nalpha += nalphaprime
        nreflect += nreflectprime
        
        # Decide if it's time to stop.
        sturn = stop_criterion(thetaminus, thetaplus, vminus, vplus)
        #print(sprime, sturn)
        s = sprime and sturn
        
        if not can_continue:
            if direction == 1:
                fwd_possible = False
            if direction == -1:
                rwd_possible = False
        
        #if not s and (fwd_possible or rwd_possible):
        #    print("U-turn found a:%d r:%d t:%d" % (alpha, nreflect, nalpha), sturn, sprime, (thetaminus, thetaplus), (vminus, vplus))
            #assert False

        # Increment depth.
        j += 1
    
    #print("jumping to:", theta)
    #print('Tree height: %d, accepts: %03.2f%%, reflects: %03.2f%%, epsilon=%g' % (j, alpha/nalpha*100, nreflect/nalpha*100, epsilon))
    return alpha, nreflect, nalpha, theta, p, logp, j

def generate_uniform_direction(d, massmatrix):
    # draw from a circle, with a little noise in amplitude
    momentum = np.random.multivariate_normal(np.zeros(d), np.dot(massmatrix, np.eye(d)))
    momentum /= (momentum**2).sum()**0.5
    return momentum


class DynamicCHMCSampler(object):
    """Dynamic Constrained Hamiltonian/Hybrid Monte Carlo technique
    
    Run a billiard ball inside the likelihood constrained. 
    The ball reflects off the constraint.
    
    The trajectory is explored in steps of stepsize epsilon.
    A No-U-turn criterion and randomized doubling of forward or backward
    steps is used to avoid repeating circular trajectories.
    Because of this, the number of steps is dynamic.
    """

    def __init__(self, ndim, nsteps, transform, loglike, gradient, delta=0.9, nudge=1.04):
        """Initialise sampler.

        Parameters
        -----------
        nsteps: int
            number of accepted steps until the sample is considered independent.
        transform: function
            called with unit cube position vector u, returns
            transformed parameter vector p.
        loglike: function
            called with transformed parameters p, returns loglikelihood
        gradient: function
            called with unit cube position vector u, returns
            gradient (dlogL/du, not just dlogL/dp)

        """
        self.history = []
        self.nsteps = nsteps
        self.scale = 0.1 * ndim**0.5
        self.transform = transform
        self.loglike = loglike
        self.gradient = transform
        self.nudge = nudge
        self.delta = delta
        self.massmatrix = 1
        self.invmassmatrix = 1
        
        self.logstat = []
        self.logstat_labels = ['acceptance_rate', 'reflect_fraction', 'stepsize', 'treeheight']
        self.logstat_trajectory = []

    def __str__(self):
        """Get string representation."""
        return type(self).__name__ + '(nsteps=%d)' % self.nsteps

    def plot(self, filename):
        """Plot sampler statistics."""
        if len(self.logstat) == 0:
            return

        parts = np.transpose(self.logstat)
        plt.figure(figsize=(10, 1 + 3 * len(parts)))
        for i, (label, part) in enumerate(zip(self.logstat_labels, parts)):
            plt.subplot(len(parts), 1, 1 + i)
            plt.ylabel(label)
            plt.plot(part)
            x = []
            y = []
            for j in range(0, len(part), 20):
                x.append(j)
                y.append(part[j:j + 20].mean())
            plt.plot(x, y)
            if np.min(part) > 0:
                plt.yscale('log')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=40, plot=False):
        """Get a new point.

        Parameters
        ----------
        region: MLFriends
            region.
        Lmin: float
            loglikelihood threshold
        us: array of vectors
            current live points
        Ls: array of floats
            current live point likelihoods
        transform: function
            transform function
        loglike: function
            loglikelihood function
        ndraw: int
            number of draws to attempt simultaneously.
        plot: bool
            whether to produce debug plots.

        """
        i = np.random.randint(len(Ls))
        #print("starting from live point %d" % i)
        self.starti = i
        ui = us[i,:]
        Li = Ls[i]
        pi = None
        assert np.logical_and(ui > 0, ui < 1).all(), ui
        
        ncalls_total = 1
        
        nsteps_remaining = self.nsteps
        while nsteps_remaining > 0:
            unew, pnew, Lnew, nc, alpha, fracreflect, treeheight = self.move(ui, pi, Li,
                region=region, ndraw=ndraw, plot=plot, Lmin=Lmin)
            
            if pnew is not None:
                # do not count failed accepts
                nsteps_remaining = nsteps_remaining - 1
            #else:
            #    print("stuck:", Li, "->", Lnew, "Lmin:", Lmin, nsteps_remaining)
            
            ncalls_total += nc
            #print(" ->", Li, Lnew, unew, pnew)
            assert np.logical_and(unew > 0, unew < 1).all(), unew
            
            if plot:
                plt.plot([ui[0], unew[:,0]], [ui[1], unew[:,1]], '-', color='k', lw=0.5)
                plt.plot(ui[0], ui[1], 'd', color='r', ms=4)
                plt.plot(unew[:,0], unew[:,1], 'x', color='r', ms=4)
            
            ui, pi, Li = unew, pnew, Lnew
        
            self.logstat_trajectory.append([alpha, fracreflect, treeheight])
        
        self.adjust_stepsize()
        
        return ui, pi, Li, ncalls_total

    def move(self, ui, pi, Li, region, Lmin, ndraw=1, plot=False):
        """Move from position ui, Li, gradi with a HMC trajectory.
        
        Return
        ------
        unew: vector
            new position in cube space
        pnew: vector
            new position in physical parameter space
        Lnew: float
            new likelihood
        nc: int
            number of likelihood evaluations
        alpha: float
            acceptance rate of trajectory
        treeheight: int
            height of NUTS tree
        """
        
        epsilon = self.scale
        epsilon_here = 10**np.random.normal(0, 0.3) * epsilon
        #epsilon_here = np.random.uniform() * epsilon
        #epsilon_here = epsilon
        d = len(ui)
        
        assert Li >= Lmin

        # draw from momentum
        v = generate_uniform_direction(d, self.massmatrix)
        
        # explore and sample from one trajectory
        alpha, nreflects, nalpha, theta, pnew, Lnew, treeheight = tree_sample(
            ui, pi, Li, v, epsilon_here, 
            self.transform, self.loglike, self.gradient, Lmin, maxheight=15)

        return theta, pnew, Lnew, nalpha, alpha / nalpha, nreflects / nalpha, treeheight

    
    def create_problem(self, Ls, region):
        """ Set up auxiliary distribution.
        
        Parameters
        ----------
        Ls: array of floats
            live point likelihoods
        region: MLFriends region object
            region.transformLayer is used to obtain mass matrices
        """
        
        # problem dimensionality
        layer = region.transformLayer

        if hasattr(layer, 'invT'):
            self.invmassmatrix = layer.cov
            self.massmatrix = np.linalg.inv(self.invmassmatrix)
        elif hasattr(layer, 'std'):
            if np.shape(layer.std) == () and layer.std == 1:
                self.massmatrix = 1
                self.invmassmatrix = 1
            else:
                # invmassmatrix: covariance
                self.invmassmatrix = np.diag(layer.std[0]**2)
                self.massmatrix = np.diag(layer.std[0]**-2)
                print(self.invmassmatrix.shape, layer.std)
        
    def adjust_stepsize(self):
        if len(self.logstat_trajectory) == 0:
            return
        
        # log averaged acceptance and trajectory statistics
        self.logstat.append([
            np.mean([alpha for alpha, fracreflect, treeheight in self.logstat_trajectory]),
            np.mean([fracreflect for alpha, fracreflect, treeheight in self.logstat_trajectory]),
            float(self.scale),
            np.mean([2**treeheight for alpha, fracreflect, treeheight in self.logstat_trajectory])
        ])
        
        #nsteps = len(self.logstat_trajectory)
        # update step size based on collected acceptance rates
        #if any([treeheight <= 1 for alpha, beta, treeheight in self.logstat_trajectory]):
        #    # stuck, no move. Finer steps needed.
        #    self.scale /= self.nudge
        #elif all([2**treeheight > 10 for alpha, beta, treeheight in self.logstat_trajectory]):
        #    # slowly go towards more efficiency
        #    self.scale *= self.nudge**(1./40)
        #else:
        #    alphamean, scale, betamean, treeheightmean = self.logstat[-1]
        #    if alphamean < self.delta:
        #        self.scale /= self.nudge
        #    elif alphamean > self.delta:
        #        self.scale *= self.nudge
        N = int(min(200 // self.nsteps, 1))
        #alphamean, reflectmean, _, treeheightmean = self.logstat[-1]
        alphamean = np.mean([alphamean for alphamean, _, _, _ in self.logstat[-N:]])
        reflectmean = np.mean([reflectmean for _, reflectmean, _, _ in self.logstat[-N:]])
        treeheightmean = np.mean([treeheightmean for _, _, _, treeheightmean in self.logstat[-N:]])
        #if treeheightmean < 4:
        #    self.scale /= self.nudge
        #elif rejectmean < 1 - self.delta:
        #    self.scale *= self.nudge

        # aim towards an acceptance rate of delta
        if alphamean > self.delta:
            self.scale *= self.nudge**(1./N)
        else:
            self.scale /= self.nudge**(1./N)
        #if self.scale < 1e-5:
        #    self.scale = 1e-5
        
        """
        for alpha, fracreflect, treeheight in self.logstat_trajectory:
            # aim towards delta acceptance rate
            # but weigh the rates from short trees more
            if alpha > self.delta:
                self.scale *= self.nudge * 2**treeheight
            else:
                self.scale /= self.nudge * 2**treeheight
        """
        self.logstat_trajectory = []
        if len(self.logstat) % N == 0:
            print("updating step size: %.4f %.4f %.1f" % (alphamean, reflectmean, treeheightmean), "-->", self.scale)

    def region_changed(self, Ls, region):
        self.adjust_stepsize()
        self.create_problem(Ls, region)
        
        
