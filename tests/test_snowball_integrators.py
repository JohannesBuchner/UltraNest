"""Agreement of the snowballingns integrators on one shared trajectory.

The same released dead points (one shared snowball trajectory) are fed
to the graph-based MultiCounter and to the custom 1/K-strip
accumulation; both must agree.
"""
import numpy as np
from numpy import logaddexp

from ultranest.netiter import MultiCounter, TreeNode
from ultranest.snowballingns.state import SnowballState


def _gaussian_problem(d=2):
    """Flat prior on [0,1]^d, Gaussian likelihood around 0.5."""

    def transform(x):
        return x

    def loglike(x):
        x = np.asarray(x, dtype=float).reshape(-1, d)
        return -0.5 * (((x - 0.5) / 0.1) ** 2).sum(axis=1)

    return transform, loglike


def run_trajectory(K=25, P=5, d=2, nbatches=40, seed=1):
    """One snowball trajectory: `nbatches` batch advances, returning the
    initial prior-draw likelihoods and the released dead points (thread,
    threshold, walk-end likelihood) in release order."""
    np.random.seed(seed)
    transform, loglike = _gaussian_problem(d)
    u = np.random.uniform(size=(K, d))
    L = np.asarray(loglike(transform(u)), dtype=float).reshape(-1)
    # set_heads aliases L (receive mutates the heads in place), so keep
    # a copy of the prior-draw likelihoods
    L_init = L.copy()
    state = SnowballState(K, P, d, nsteps=1, seed=seed + 1,
                          integrate=False)
    state.set_heads(u, L)
    state.region = None
    state.stepsampler = None
    released = []
    for _ in range(nbatches):
        state.start_round()
        while state.any_active():
            prop = state.propose()
            pnew = transform(prop)
            Lnew = np.asarray(loglike(pnew), dtype=float).reshape(-1)
            state.receive(Lnew)
        threads, values, rows = state.release()
        if len(threads):
            data = state.store.read(rows)
            walkend = np.asarray(data[:, 1], dtype=float)
            released.extend(zip(threads.tolist(), values.tolist(),
                                walkend.tolist()))
    return state, L_init, released


def test_custom_multicounter_agreement():
    state, L_init, released = run_trajectory()
    K = state.K
    assert len(released) > 0
    thresholds = np.asarray([v for _, v, _ in released])
    # the release order is the threshold order
    assert np.all(np.diff(thresholds) >= 0)

    # the graph-based MultiCounter: the sequences are branches of a
    # tree, every dead point extends its thread's branch
    main = MultiCounter(nroots=K, nbootstraps=0, random=False)
    _, headL = state.get_heads()
    headL = np.asarray(headL, dtype=float)
    rootids = np.arange(K)
    thread_nodes = [TreeNode(value=float(l), id=-1) for l in L_init]
    for j, threshold, walkend in released:
        dead_node = thread_nodes[j]
        child = TreeNode(value=float(walkend), id=-1)
        dead_node.children.append(child)
        thread_nodes[j] = child
        main.passing_node(j, dead_node, rootids, headL)

    # the custom integrator: 1/K strips in threshold order
    logz = -np.inf
    logvrem = 0.0
    step = float(np.log1p(-np.exp(-1.0 / K)))
    for _, threshold, _ in released:
        logwidth = logvrem + step
        logz = logaddexp(logz, logwidth + threshold)
        logvrem -= 1.0 / K

    assert len(main.logweights) == len(released)
    assert abs(main.logZ - logz) < 1e-8
    assert abs(main.logVolremaining - logvrem) < 1e-8


def test_kfolding_fold_mean_tracks_main():
    """The K fold integrators (one per sequence, 1-live-point runs)
    estimate the same integral as the main integrator."""
    state, L_init, released = run_trajectory()
    K = state.K
    main = MultiCounter(nroots=K, nbootstraps=0, random=False)
    folds = [MultiCounter(nroots=1, nbootstraps=0, random=False)
             for _ in range(K)]
    rootids = np.arange(K)
    fold_rootids = np.arange(1)
    thread_nodes = [TreeNode(value=float(l), id=-1) for l in L_init]
    _, headL = state.get_heads()
    headL = np.asarray(headL, dtype=float)
    for j, threshold, walkend in released:
        dead_node = thread_nodes[j]
        child = TreeNode(value=float(walkend), id=-1)
        dead_node.children.append(child)
        thread_nodes[j] = child
        main.passing_node(j, dead_node, rootids, headL)
        folds[j].passing_node(0, dead_node, fold_rootids, headL)
    logz_folds = np.array([f.logZ for f in folds])
    finite = np.isfinite(logz_folds)
    # most sequences contributed at least one dead point
    assert finite.sum() > K / 2
    err_folds = float(np.sqrt(np.var(logz_folds[finite]) / finite.sum()))
    assert err_folds > 0
    # the fold mean estimates the same integral as the main integrator
    assert abs(logz_folds[finite].mean() - main.logZ) < 5 * err_folds + 0.2
