'''Diagnostic: feed the same dead points to the custom integrator, the
MultiCounter and the SingleCounter on one trajectory, comparing the
integrated logZ per rotation. Usage: python diagnose_integrators.py [frac_remain]'''
import sys
sys.path.insert(0, 'evaluate')
import numpy as np
from numpy import log
from ultranest.netiter import TreeNode, MultiCounter, SingleCounter
from ultranest.snowballingns.state import SnowballState
from ultranest.snowballingns.integrator import SnowballingNestedSampler, make_region
from ultranest.perfectns import BananaGaussianLRPS

frac_remain = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
problem = BananaGaussianLRPS.standard_problem(2)
transform, loglike = problem.transform, problem.loglike
d = len(problem.paramnames)

np.random.seed(1)
K = 100
u = np.random.uniform(size=(K, d))
L = np.asarray(loglike(transform(u)), dtype=float).reshape(-1)

state = SnowballState(K, 10, d, nsteps=1, seed=2, integrate=True)
state.set_heads(u, L)
sampler = SnowballingNestedSampler.__new__(SnowballingNestedSampler)
sampler.paramnames = problem.paramnames
sampler.x_dim = d
sampler.loglike = loglike
sampler.transform = transform
sampler.state = state
sampler.K = K
sampler.root = None
sampler.thread_nodes = []
sampler.chain_nodes = []
sampler.Lmax_global = float(np.max(L))
sampler.stepsampler = None
sampler.region_class = __import__('ultranest.mlfriends', fromlist=['RobustEllipsoidRegion']).RobustEllipsoidRegion
sampler.num_bootstraps = 10
sampler.results = None
sampler.rotation_summaries = []
sampler.nlive_per_rotation = []
sampler.nrotations = 0
sampler.previous_nodes = []
sampler._rotation_new_nodes = []
sampler._prev_ptr = 0
sampler._store_rows = []
sampler._logweights = []
sampler._logvols = []
sampler.bin_edges = None

region = sampler._make_region(u)
state.set_region(region)
state.stepsampler = sampler.stepsampler
sampler.root = TreeNode(value=-np.inf, id=-1)
for t in range(K):
    node = TreeNode(value=float(L[t]), id=-1)
    sampler.root.children.append(node)
    sampler.thread_nodes.append(node)

print('frac_remain = %g; analytic lnZ = -2.7693' % frac_remain)
rootids = np.arange(K)
for rotation in range(1, 6):
    main_iterator = MultiCounter(nroots=sampler.K, nbootstraps=10,
                                 random=False, check_insertion_order=False)
    single_iterator = SingleCounter(random=False)
    sampler.main_iterator = main_iterator
    sampler._prev_ptr = 0
    while True:
        state.start_round()
        while state.any_active():
            prop = state.propose()
            pnew = transform(prop)
            Lnew = np.asarray(loglike(pnew), dtype=float).reshape(-1)
            state.receive(Lnew)
        _, headL = state.get_heads()
        front = float(np.min(headL))
        threads, values, rows = state.release()
        if len(threads):
            data = state.store.read(rows)
            walkend_L = np.asarray(data[:, 1], dtype=float)
        else:
            walkend_L = np.zeros(0)
        old_cands = []
        while sampler._prev_ptr < len(sampler.previous_nodes) \
                and sampler.previous_nodes[sampler._prev_ptr][0].value < front:
            old_cands.append(sampler.previous_nodes[sampler._prev_ptr])
            sampler._prev_ptr += 1
        new_cands = sorted(zip(threads, values, rows, walkend_L),
                           key=lambda c: c[1])
        merged = []
        a = b = 0
        while a < len(old_cands) and b < len(new_cands):
            if old_cands[a][0].value < new_cands[b][1]:
                merged.append(('old', old_cands[a])); a += 1
            else:
                merged.append(('new', new_cands[b])); b += 1
        merged += [('old', c) for c in old_cands[a:]] \
                + [('new', c) for c in new_cands[b:]]
        for kind, c in merged:
            if kind == 'old':
                node, rootid = c
                main_iterator.passing_node(rootid, node, rootids,
                                           np.asarray(headL, dtype=float))
                single_iterator.passing_node(node, list(sampler.thread_nodes))
                row, t = node.id, rootid
            else:
                t, value, row, walkend = c
                dead_node = sampler.thread_nodes[t]
                child = TreeNode(value=float(walkend), id=int(row))
                dead_node.children.append(child)
                sampler.thread_nodes[t] = child
                sampler.chain_nodes.append((child, int(t)))
                sampler._rotation_new_nodes.append((child, int(t)))
                main_iterator.passing_node(int(t), dead_node, rootids,
                                           np.asarray(headL, dtype=float))
                single_iterator.passing_node(dead_node,
                                             list(sampler.thread_nodes))
            x = -main_iterator.logVolremaining
            sampler._store_rows.append((row, x, t))
            sampler._logweights.append(
                float(main_iterator.logweights[-1][0]))
            sampler._logvols.append(x)
        sampler.Lmax_global = max(sampler.Lmax_global, float(np.max(headL)))
        single_iterator.Lmax = sampler.Lmax_global
        logzremain = sampler.Lmax_global + main_iterator.logVolremaining
        if logzremain - main_iterator.logZ < log(frac_remain):
            break
    _, headL = state.get_heads()
    logz_custom = float(np.logaddexp(
        state.logZ, state.logVrem + np.logaddexp.reduce(headL)
        - np.log(sampler.K)))
    logz_mc = float(np.logaddexp(main_iterator.logZ, main_iterator.logZremain))
    single_iterator.Lmax = sampler.Lmax_global
    logz_sc = float(np.logaddexp(single_iterator.logZ,
                                 single_iterator.logZremain))
    print('rot %d: K=%4d custom %+.4f  multicnt %+.4f  single %+.4f   '
          '(logZ int: cust %+.4f mc %+.4f sc %+.4f; logZremain mc %+.4f '
          'sc %+.4f)'
          % (rotation, sampler.K, logz_custom, logz_mc, logz_sc,
             state.logZ, main_iterator.logZ, single_iterator.logZ,
             main_iterator.logZremain, single_iterator.logZremain),
          flush=True)
    K_old = sampler.K
    u_add = np.random.uniform(size=(K_old, d))
    L_add = np.asarray(loglike(transform(u_add)), dtype=float).reshape(-1)
    state.widen(u_add, L_add)
    for t in range(len(u_add)):
        node = TreeNode(value=float(L_add[t]), id=-1)
        sampler.root.children.append(node)
        sampler.thread_nodes.append(node)
        sampler.chain_nodes.append((node, K_old + t))
    sampler.K = 2 * K_old
