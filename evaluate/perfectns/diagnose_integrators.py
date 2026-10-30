import sys
sys.path.insert(0, 'evaluate')
import numpy as np
from numpy import log
from ultranest.netiter import TreeNode, MultiCounter, SingleCounter
from ultranest.snowballingns.state import SnowballState
from ultranest.snowballingns.integrator import SnowballingNestedSampler, make_region
from ultranest.perfectns import BananaGaussianLRPS

frac_remain = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
problem = BananaGaussianLRPS.standard_problem(2)
transform, loglike = problem.transform, problem.loglike
d = len(problem.paramnames)
print('frac_remain = %g, seeds = %d; analytic lnZ = -2.7693'
      % (frac_remain, nseeds))

for seed in range(1, nseeds + 1):
    np.random.seed(seed)
    K = 100
    u = np.random.uniform(size=(K, d))
    L = np.asarray(loglike(transform(u)), dtype=float).reshape(-1)
    state = SnowballState(K, 10, d, nsteps=1, seed=seed + 1,
                          integrate=False)
    state.set_heads(u, L)
    sampler = SnowballingNestedSampler.__new__(SnowballingNestedSampler)
    sampler.paramnames = problem.paramnames
    sampler.x_dim = d
    sampler.loglike = loglike
    sampler.transform = transform
    sampler.state = state
    sampler.K = K
    sampler.Lmax_global = float(np.max(L))
    sampler.stepsampler = None
    sampler.region_class = __import__(
        'ultranest.mlfriends', fromlist=['RobustEllipsoidRegion']
    ).RobustEllipsoidRegion
    sampler.num_bootstraps = 10
    sampler.previous_nodes = []
    sampler._rotation_new_nodes = []
    sampler.chain_nodes = []
    sampler._prev_ptr = 0
    sampler._store_rows = []
    sampler._logweights = []
    sampler._logvols = []
    sampler.bin_edges = None
    region = sampler._make_region(u)
    state.set_region(region)
    state.stepsampler = sampler.stepsampler
    sampler.root = TreeNode(value=-np.inf, id=-1)
    thread_nodes = []
    for t in range(K):
        node = TreeNode(value=float(L[t]), id=-1)
        sampler.root.children.append(node)
        thread_nodes.append(node)
    sampler.thread_nodes = thread_nodes

    for rotation in range(1, 5):
        # mirror run_iter: the dead points of the earlier rotations
        # (registered at their deaths) enter the re-walk set
        sampler.previous_nodes.extend(sampler._rotation_new_nodes)
        sampler.previous_nodes.sort(key=lambda c: c[0].value)
        sampler._rotation_new_nodes = []
        sampler._prev_ptr = 0
        main_iterator = MultiCounter(nroots=sampler.K, nbootstraps=10,
                                     random=False,
                                     check_insertion_order=False)
        single_iterator = SingleCounter(random=False)
        sampler.main_iterator = main_iterator
        cust_logZ = -np.inf; cust_logVrem = 0.0
        newonly_logZ = -np.inf; newonly_logVrem = 0.0
        step = float(np.log1p(-np.exp(-1.0 / sampler.K)))
        invK = 1.0 / sampler.K
        first_divergence = None
        while True:
            state.start_round()
            while state.any_active():
                prop = state.propose()
                pnew = transform(prop)
                Lnew = np.asarray(loglike(pnew), dtype=float).reshape(-1)
                state.receive(Lnew)
            _, headL = state.get_heads()
            front = float(np.min(headL))
            headL_f = np.asarray(headL, dtype=float)
            rootids = np.arange(sampler.K)
            threads, values, rows = state.release()
            if len(threads):
                data = state.store.read(rows)
                walkend_L = np.asarray(data[:, 1], dtype=float)
            else:
                walkend_L = np.zeros(0)
            old_cands = []
            while sampler._prev_ptr < len(sampler.previous_nodes) \
                    and sampler.previous_nodes[sampler._prev_ptr][0].value \
                    < front:
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
                    dead_node, t = c
                    row = dead_node.id
                    Lc = dead_node.value
                else:
                    t, value, row, walkend = c
                    Lc = float(value)
                    dead_node = thread_nodes[t]
                    child = TreeNode(value=float(walkend), id=int(row))
                    dead_node.children.append(child)
                    thread_nodes[t] = child
                    sampler.chain_nodes.append((child, int(t)))
                    # the dead point enters the re-walk set at its death
                    sampler._rotation_new_nodes.append((dead_node, int(t)))
                # MultiCounter
                main_iterator.passing_node(int(t), dead_node, rootids,
                                           headL_f)
                # SingleCounter
                single_iterator.passing_node(dead_node, list(thread_nodes))
                # the custom inline accumulators
                logwidth = cust_logVrem + step
                cust_logZ = np.logaddexp(cust_logZ, logwidth + Lc)
                cust_logVrem -= invK
                if kind == 'new':
                    newonly_logZ = np.logaddexp(
                        newonly_logZ, newonly_logVrem + step + Lc)
                    newonly_logVrem -= invK
                if first_divergence is None and abs(
                        main_iterator.logVolremaining - cust_logVrem) > 1e-9:
                    first_divergence = dict(
                        kind=kind, L=float(Lc),
                        mc=float(main_iterator.logVolremaining),
                        cust=float(cust_logVrem))
            sampler.Lmax_global = max(sampler.Lmax_global,
                                      float(np.max(headL)))
            single_iterator.Lmax = sampler.Lmax_global
            logzremain = sampler.Lmax_global + main_iterator.logVolremaining
            if logzremain - main_iterator.logZ < log(frac_remain):
                break
        # widen: double K with fresh prior draws
        K_old = sampler.K
        u_add = np.random.uniform(size=(K_old, d))
        L_add = np.asarray(loglike(transform(u_add)), dtype=float).reshape(-1)
        state.widen(u_add, L_add)
        for t in range(len(u_add)):
            node = TreeNode(value=float(L_add[t]), id=-1)
            sampler.root.children.append(node)
            thread_nodes.append(node)
        sampler.K = 2 * K_old
        print('seed %d rot %d: K=%4d | logZ_int: mc %+.4f sc %+.4f '
              'cust %+.4f newonly %+.4f | reported: mc %+.4f sc %+.4f '
              'cust %+.4f newonly %+.4f | first divergence: %s'
              % (seed, rotation, sampler.K,
                 main_iterator.logZ, single_iterator.logZ, cust_logZ,
                 newonly_logZ,
                 float(np.logaddexp(main_iterator.logZ, sampler.Lmax_global
                                    + main_iterator.logVolremaining)),
                 float(np.logaddexp(single_iterator.logZ,
                                    sampler.Lmax_global
                                    + single_iterator.logVolremaining)),
                 float(np.logaddexp(cust_logZ, sampler.Lmax_global
                                    + cust_logVrem)),
                 float(np.logaddexp(newonly_logZ, sampler.Lmax_global
                                    + newonly_logVrem)),
                 str(first_divergence)), flush=True)
    print()
