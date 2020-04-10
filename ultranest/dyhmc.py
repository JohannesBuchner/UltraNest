"""Experimental constrained Hamiltanean Monte Carlo step sampling

Contrary to CHMC, this uses the likelihood gradients throughout the path.
A helper surface is created using the live points.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.stats


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
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def leapfrog(theta, r, grad, epsilon, invmassmatrix, f):
    """Leap frog step from theta with momentum r and stepsize epsilon.
    The local gradient grad is updated with function f"""
    # make half step in r
    rprime = r + 0.5 * epsilon * grad
    # make new step in theta
    thetaprime = theta + epsilon * np.dot(invmassmatrix, rprime)
    # compute new gradient
    (logpprime, gradprime), extra = f(thetaprime)
    # make half step in r again
    rprime = rprime + 0.5 * epsilon * gradprime
    return thetaprime, rprime, gradprime, logpprime, extra


def build_tree(theta, r, grad, v, j, epsilon, invmassmatrix, f, joint0):
    """The main recursion."""
    if j == 0:
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, rprime, gradprime, logpprime, extraprime = leapfrog(theta, r, grad, v * epsilon, invmassmatrix, f)
        joint = logpprime - 0.5 * np.dot(np.dot(rprime, invmassmatrix), rprime.T)
        # Is the simulation wildly inaccurate?
        sprime = joint0 - 1000. < joint  # and logpprime > Lmin
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        r = rprime[:]
        gradminus = gradprime[:]
        gradplus = gradprime[:]
        # Compute the acceptance probability.
        if not sprime:
            # print("stopped trajectory:", joint0, joint, logpprime, gradprime)
            alphaprime = 0.0
        else:
            alphaprime = min(1., np.exp(joint - joint0))

        if logpprime < -300:
            # if alphaprime > 0:
            #    print("stopping at very low probability:", joint0, joint, logpprime, gradprime)
            betaprime = 0.0
        else:
            betaprime = alphaprime * np.exp(-logpprime)

        if betaprime == 0.0:
            sprime = False

        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, \
            thetaprime, gradprime, logpprime, extraprime, rprime, sprime, \
            alphaprime, betaprime, nalphaprime = build_tree(
                theta, r, grad, v, j - 1, epsilon, invmassmatrix, f, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if sprime:
            if v == -1:
                thetaminus, rminus, gradminus, _, _, _, \
                    thetaprime2, gradprime2, logpprime2, extraprime2, \
                    rprime2, sprime2, alphaprime2, betaprime2, nalphaprime2 = build_tree(
                        thetaminus, rminus, gradminus, v, j - 1, epsilon, invmassmatrix, f, joint0)
            else:
                _, _, _, thetaplus, rplus, gradplus, \
                    thetaprime2, gradprime2, logpprime2, extraprime2, \
                    rprime2, sprime2, alphaprime2, betaprime2, nalphaprime2 = build_tree(
                        thetaplus, rplus, gradplus, v, j - 1, epsilon, invmassmatrix, f, joint0)

            # Choose which subtree to propagate a sample up from.
            if betaprime + betaprime2 > 0 and np.random.uniform() < betaprime2 / (betaprime + betaprime2):
                thetaprime = thetaprime2[:]
                gradprime = gradprime2[:]
                logpprime = logpprime2
                extraprime = extraprime2
                rprime = rprime2

            # Update the stopping criterion.
            sturn = stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # print(sprime, sprime2, sturn)
            sprime = sprime and sprime2 and sturn
            # Update the acceptance probability statistics.
            alphaprime += alphaprime2
            betaprime += betaprime2
            nalphaprime += nalphaprime2

    return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, \
        thetaprime, gradprime, logpprime, extraprime, \
        rprime, sprime, alphaprime, betaprime, nalphaprime


def tree_sample(theta, logp, r0, grad, extra, epsilon, invmassmatrix, f, joint, maxheight=np.inf):
    """Build NUTS-like tree of sampling path from theta towards p with stepsize epsilon."""
    # initialize the tree
    thetaminus = theta
    thetaplus = theta
    rminus = r0[:]
    rplus = r0[:]
    gradminus = grad[:]
    gradplus = grad[:]
    alpha = 1
    beta = 1
    nalpha = 1

    j = 0  # initial heigth j = 0
    s = True  # Main loop: will keep going until s == 0.

    while s and j < maxheight:
        # Choose a direction. -1 = backwards, 1 = forwards.
        v = int(2 * (np.random.uniform() < 0.5) - 1)

        # Double the size of the tree.
        if v == -1:
            thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, \
                logpprime, extraprime, rprime, sprime, \
                alphaprime, betaprime, nalphaprime = build_tree(
                    thetaminus, rminus, gradminus, v, j, epsilon, invmassmatrix, f, joint)
        else:
            _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, \
                logpprime, extraprime, rprime, sprime, \
                alphaprime, betaprime, nalphaprime = build_tree(
                    thetaplus, rplus, gradplus, v, j, epsilon, invmassmatrix, f, joint)

        assert beta > 0, beta
        assert betaprime >= 0, betaprime

        # Use Metropolis-Hastings to decide whether or not to move to a
        # point from the half-tree we just generated.
        if sprime and np.random.uniform() < betaprime / (beta + betaprime):
            logp = logpprime
            grad = gradprime[:]
            theta = thetaprime
            extra = extraprime
            r0 = rprime
            # print("accepting", theta, logp)

        alpha += alphaprime
        beta += betaprime
        nalpha += nalphaprime

        # Decide if it's time to stop.
        sturn = stop_criterion(thetaminus, thetaplus, rminus, rplus)
        # print(sprime, sturn)
        s = sprime and sturn
        # Increment depth.
        j += 1
    # print("jumping to:", theta)
    # print('Tree height: %d, acceptance fraction: %03.2f%%/%03.2f%%, epsilon=%g' % (j, alpha/nalpha*100, beta/nalpha*100, epsilon))
    return alpha, beta, nalpha, theta, grad, logp, extra, r0, j


def find_beta_params_static(d, u10):
    """ Define auxiliary distribution following naive intuition.
    Make 50% quantile to be at u=0.1, and very flat at high u. """
    del d
    betas = np.arange(1, 20)
    z50 = scipy.special.betaincinv(1.0, betas, 0.5)

    alpha = 1
    beta = np.interp(u10, z50[::-1], betas[::-1])
    print("Auxiliary Beta distribution(alpha=%.1f, beta=%.1f)" % (alpha, beta))
    return alpha, beta


def find_beta_params_dynamic(d, u10):
    """ Define auxiliary distribution taking into account
    kinetic energy of a d-dimensional HMC.
    Make exp(-d/2) quantile to be at u=0.1, and 95% quantile at u=0.5. """
    del d

    u50 = (u10 + 1) / 2.

    def minfunc(params):
        """ minimization function """
        alpha, beta = params
        q10 = scipy.special.betainc(alpha, beta, u10)
        q50 = scipy.special.betainc(alpha, beta, u50)
        return (q10 - np.exp(-d / 2))**2 + (q50 - 0.98)**2

    r = scipy.optimize.minimize(minfunc, [1.0, 10.0])
    alpha, beta = r.x
    print("Auxiliary Beta distribution(alpha=%.1f, beta=%.1f)" % (alpha, beta), u10)

    return alpha, beta


def generate_momentum_normal(d, massmatrix):
    """ draw direction vector according to mass matrix """
    return np.random.multivariate_normal(np.zeros(d), np.dot(massmatrix, np.eye(d)))


def generate_momentum(d, massmatrix, alpha, beta):
    """ draw momentum from a circle, with amplitude following the beta distribution """
    momentum = np.random.multivariate_normal(np.zeros(d), np.dot(massmatrix, np.eye(d)))
    # generate normalisation from beta distribution
    # add a bit of noise in the step size
    # norm *= np.uniform(0.2, 2)
    betainc = scipy.special.betainc
    auxnorm = -betainc(alpha + 1, beta, 1) + betainc(alpha + 1, beta, 0) + betainc(alpha, beta, 1)
    u = np.random.uniform()
    if u > 0.9:
        norm = 1.
    else:
        u /= 0.9
        norm = betainc(alpha, beta, u)
    momnorm = -np.log((norm + 1e-10) / auxnorm)
    assert momnorm >= 0, (momnorm, norm, auxnorm)

    momentum *= momnorm / (momentum**2).sum()**0.5

    return momentum


def generate_momentum_circle(d, massmatrix):
    """ draw from a circle, with a little noise in amplitude """
    momentum = np.random.multivariate_normal(np.zeros(d), np.dot(massmatrix, np.eye(d)))
    momentum *= 10**np.random.uniform(-0.3, 0.3) / (momentum**2).sum()**0.5
    return momentum


def generate_momentum_flattened(d, massmatrix):
    """ like normal distribution, but make momenta distributed like a single gaussian.
    **this is the one being used** """
    momentum = np.random.multivariate_normal(np.zeros(d), np.dot(massmatrix, np.eye(d)))
    norm = (momentum**2).sum()**0.5
    assert norm > 0
    momentum *= norm**(1 / d) / norm
    return momentum


class FlattenedProblem(object):
    """
    Creates a suitable auxiliary distribution from samples of likelihood values

    The distribution is the CDF of a beta distribution, with
    0 -> logLmin
    1 -> 90% quantile of logLs
    0.5 -> 10% quantile of logLs

    .modify_Lgrad() returns the conversion from logL, grad to the
    equivalents on the auxiliary distribution.

    .__call__(x) returns logL, grad on the auxiliary distribution.
    """

    def __init__(self, d, Ls, function, layer):
        self.Lmin = Ls.min()
        self.L90 = np.percentile(Ls, 90)
        self.L10 = np.percentile(Ls, 10)
        u10 = (self.L10 - self.Lmin) / (self.L90 - self.Lmin)

        self.function = function
        self.layer = layer

        # self.alpha, self.beta = find_beta_params_static(d, u10)
        # self.alpha, self.beta = find_beta_params_dynamic(d, u10)
        self.alpha, self.beta = 1.0, 6.0

        self.du_dL = 1 / (self.L90 - self.Lmin)
        # print("du/dL = %g " % du_dL)
        self.C = scipy.special.beta(self.alpha, self.beta)
        self.d = d

        if hasattr(self.layer, 'invT'):
            self.invmassmatrix = self.layer.cov
            self.massmatrix = np.linalg.inv(self.invmassmatrix)
            # print("invM:", self.invmassmatrix.shape)
        elif hasattr(self.layer, 'std'):
            if np.shape(self.layer.std) == () and self.layer.std == 1:
                self.massmatrix = 1
                self.invmassmatrix = 1
            else:
                # invmassmatrix: covariance
                self.invmassmatrix = np.diag(self.layer.std[0]**2)
                self.massmatrix = np.diag(self.layer.std[0]**-2)
                print(self.invmassmatrix.shape, self.layer.std)
        else:
            assert False

    def modify_Lgrad(self, L, grad):
        u = (L - self.Lmin) / (self.L90 - self.Lmin)
        if u <= 0:
            logp = -np.inf
            u = 0.0
            dlogp_du = 1.0
            # print("L <= Lmin", L, self.Lmin)
        elif u > 1:
            u = 1.0
            p = 1.0
            logp = 0.0
            # print("L > L90", L, L90)
            return logp, 0 * grad
        else:
            # p = self.rv.cdf(u)
            p = scipy.special.betainc(self.alpha, self.beta, u)
            logp = np.log(p)
            B = p * self.C
            dlogp_du = u**(self.alpha - 1) * (1 - u)**(self.beta - 1) / B

        # convert gradient to flattened space
        tgrad = grad * dlogp_du * self.du_dL

        return logp, tgrad

    def __call__(self, u):
        if not np.logical_and(u > 0, u < 1).all():
            # outside unit cube, invalid.
            # print("outside", u)
            return (-np.inf, 0. * u), (None, -np.inf, 0. * u)

        p, L, grad_orig = self.function(u)
        # print("at ", u, "L:", L)
        return self.modify_Lgrad(L, grad_orig), (p, L, grad_orig)

    def generate_momentum(self):
        return generate_momentum_flattened(self.d, self.massmatrix)
        return generate_momentum_normal(self.d, self.massmatrix)
        return generate_momentum(self.d, self.massmatrix, self.alpha, self.beta)


class DynamicHMCSampler(object):
    """Dynamic Hamiltonian/Hybrid Monte Carlo technique

    Typically, HMC operates on the posterior. It has the benefit
    of producing "orbit" trajectories, that can follow the guidance
    of gradients.

    In nested sampling, we need to sample the prior subject to the
    likelihood constraint. This means a HMC would most of the time
    go in straight lines, until it steps outside the boundary.
    Techniques such as Constrained HMC and Galilean MC use the
    gradient outside to select the reflection direction.

    However, it would be beneficial to be repelled by the likelihood
    boundary, and to take advantage of gradient guidance.
    This implements a new technique that does this.

    The trick is to define a auxiliary distribution from the likelihood,
    generate HMC trajectories from it, and draw points from the
    trajectory with inverse the probability of the auxiliary distribution
    to sample from the prior. Thus, the auxiliary distribution should be
    mostly flat, and go to zero at the boundaries to repell the HMC.

    Given Lmin and Lmax from the live points,
    use a beta approximation of log-likelihood

    p=1 if L>Lmin
    u = (L - Lmin) / (Lmax - Lmin)
    p = Beta_PDF(u; alpha, beta)

    then define
     d log(p) / dx = dlog(p_orig)/dlog(p) * dlog(p_orig) / dx
     new gradient  = conversion           * original gradient

    with conversion
     dlogp/du = 0 if u>1; otherwise:
     dlogp/du =  u**(1-alpha) * (1-u)**(1-beta) / Ic(u; alpha, beta) / Beta_PDF(u, alpha, beta)
     du/dL = 1 / (Lmax - Lmin)

    The beta distribution achieves:
    * a flattening of the loglikelihood to avoid seeing only "walls"
    * using the gradient to identify how to orbit the likelihood contour
    * at higher, unseen likelihoods, the exploration is in straight lines
    * trajectory do not have the energy to go below Lmin.
    * alpha and beta parameters allow flexible choice of "contour avoidance"

    Run HMC trajectory on p
    This will draw samples proportional to p
    Modify multinomial acceptance by 1/p to get uniform samples.
    and  reject porig < p_1

    The remaining choices for HMC are how long the trajectories should
    run (number of steps) and the step size. The former is solved
    by No-U-Turn Sampler or dynamic HMC, which randomly build
    forward and backward paths until the trajectory turns around.
    Then, a random point from the trajectory is chosen.

    The step size is chosen by targeting an acceptance rate of
    delta~0.95, and decreasing(increasing) every time the region is
    rebuilt if the acceptance rate is below(above).
    """

    def __init__(self, ndim, nsteps, transform_loglike_gradient, delta=0.90, nudge=1.04):
        """Initialise sampler.

        Parameters
        -----------
        nsteps: int
            number of accepted steps until the sample is considered independent.
        transform_loglike_gradient: function
            called with unit cube position vector u, returns
            transformed parameter vector p,
            loglikelihood and gradient (dlogL/du, not just dlogL/dp)

        """
        self.history = []
        self.nsteps = nsteps
        self.nrejects = 0
        self.scale = 0.1 * ndim**0.5
        self.last = None, None, None, None
        self.transform_loglike_gradient = transform_loglike_gradient
        self.nudge = nudge
        self.delta = delta
        self.problem = None

        self.logstat = []
        self.logstat_labels = ['acceptance_rate', 'acceptance_rate_bias', 'stepsize', 'treeheight']
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
        # i = np.argsort(Ls)[0]
        mask = Ls > Lmin
        i = np.random.randint(mask.sum())
        # print("starting from live point %d" % i)
        self.starti = np.where(mask)[0][i]
        ui = us[mask,:][i]
        assert np.logical_and(ui > 0, ui < 1).all(), ui

        if self.problem is None:
            self.create_problem(Ls, region)

        ncalls_total = 1
        (Lflat, gradflat), (pi, Li, gradi) = self.problem(ui)
        assert np.shape(Lflat) == (), (Lflat, Li, gradi)
        assert np.shape(gradflat) == (len(ui),), (gradi, gradflat)

        nsteps_remaining = self.nsteps
        while nsteps_remaining > 0:
            unew, pnew, Lnew, gradnew, Lflatnew, gradflatnew, nc, alpha, beta, treeheight = self.move(
                ui, pi, Li, gradi, gradflat=gradflat, Lflat=Lflat, region=region, ndraw=ndraw, plot=plot)

            if treeheight > 1:
                # do not count failed accepts
                nsteps_remaining = nsteps_remaining - 1
            else:
                print("stuck:", Li, "->", Lnew, "Lmin:", Lmin)

            ncalls_total += nc
            # print(" ->", Li, Lnew, unew)
            assert np.logical_and(unew > 0, unew < 1).all(), unew

            if plot:
                plt.plot([ui[0], unew[:,0]], [ui[1], unew[:,1]], '-', color='k', lw=0.5)
                plt.plot(ui[0], ui[1], 'd', color='r', ms=4)
                plt.plot(unew[:,0], unew[:,1], 'x', color='r', ms=4)

            ui, pi, Li, gradi, Lflat, gradflat = unew, pnew, Lnew, gradnew, Lflatnew, gradflatnew

            self.logstat_trajectory.append([alpha, beta, treeheight])

        self.adjust_stepsize()

        return unew, pnew, Lnew, nc

    def move(self, ui, pi, Li, gradi, region, ndraw=1, Lflat=None, gradflat=None, plot=False):
        """Move from position ui, Li, gradi with a HMC trajectory.

        Return
        ------
        unew: vector
            new position in cube space
        pnew: vector
            new position in physical parameter space
        Lnew: float
            new likelihood
        gradnew: vector
            new gradient
        Lflat: float
            new likelihood on auxiliary distribution
        gradflat: vector
            new gradient on auxiliary distribution
        nc: int
            number of likelihood evaluations
        alpha: float
            acceptance rate of HMC trajectory
        beta: float
            acceptance rate of inverse-beta-biased HMC trajectory
        """

        epsilon = self.scale
        # epsilon_here = 10**np.random.normal(0, 0.3) * epsilon
        epsilon_here = np.random.uniform() * epsilon
        # epsilon_here = epsilon

        problem = self.problem
        d = len(ui)

        assert Li > problem.Lmin

        # get initial likelihood and gradient from auxiliary distribution
        if Lflat is None or gradflat is None:
            Lflat, gradflat = problem.modify_Lgrad(Li, gradi)
        assert np.shape(Lflat) == (), (Lflat, Li, gradi)
        assert np.shape(gradflat) == (d,), (gradi, gradflat)

        # draw from momentum
        momentum = problem.generate_momentum()

        # compute current Hamiltonian
        joint0 = Lflat - 0.5 * np.dot(np.dot(momentum, problem.invmassmatrix), momentum.T)
        assert np.isfinite(joint0), (
            Lflat, momentum, -0.5 * np.dot(np.dot(momentum, problem.invmassmatrix), momentum.T))

        # explore and sample from one trajectory
        alpha, beta, nalpha, theta, gradflat, Lflat, (pnew, Lnew, gradnew), rprime, treeheight = tree_sample(
            ui, Lflat, momentum, gradflat, (pi, Li, gradi), epsilon_here, problem.invmassmatrix, problem, joint0, maxheight=30)

        return theta, pnew, Lnew, gradnew, Lflat, gradflat, nalpha, alpha / nalpha, beta / nalpha, treeheight

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
        d = len(region.u[0])

        self.problem = FlattenedProblem(d, Ls, self.transform_loglike_gradient, region.transformLayer)

    def adjust_stepsize(self):
        if len(self.logstat_trajectory) == 0:
            return

        # log averaged acceptance and trajectory statistics
        self.logstat.append([
            np.mean([alpha for alpha, beta, treeheight in self.logstat_trajectory]),
            float(self.scale),
            np.mean([beta for alpha, beta, treeheight in self.logstat_trajectory]),
            np.mean([treeheight for alpha, beta, treeheight in self.logstat_trajectory])
        ])

        nsteps = len(self.logstat_trajectory)
        # update step size based on collected acceptance rates
        if any([treeheight <= 1 for alpha, beta, treeheight in self.logstat_trajectory]):
            # stuck, no move. Finer steps needed.
            self.scale /= self.nudge
        elif all([2**treeheight > 10 for alpha, beta, treeheight in self.logstat_trajectory]):
            # slowly go towards more efficiency
            self.scale *= self.nudge**(1. / 40)
        else:
            alphamean, scale, betamean, treeheightmean = self.logstat[-1]
            if alphamean < self.delta:
                self.scale /= self.nudge
            elif alphamean > self.delta:
                self.scale *= self.nudge

        self.logstat_trajectory = []
        print("updating step size: %.4f %g %.4f %.1f" % tuple(self.logstat[-1]), "-->", self.scale)

    def region_changed(self, Ls, region):
        self.adjust_stepsize()
        self.create_problem(Ls, region)
