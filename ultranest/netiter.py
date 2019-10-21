"""Functions and classes for treating nested sampling exploration as a tree.

The root represents the prior volume, branches and sub-branches split the volume.
The leaves of the tree are the integration tail.

Nested sampling proceeds as a breadth first graph search,
with active nodes sorted by likelihood value.
The number of live points are the number of parallel edges (active nodes to do).

Most functions receive the argument "roots", which are the
children of the tree root (main branches).

The exploration is bootstrap-capable without requiring additional
computational effort: The roots are indexed, and the bootstrap explorer
can ignore the rootids it does not know about.


"""


import numpy as np
from numpy import log, log1p, exp, logaddexp
import math
import operator
import sys


class TreeNode(object):
    """Simple tree node."""

    def __init__(self, value=None, id=None, children=None):
        """Define TreeNode.

        Parameters
        ----------
        value:
            used to order nodes
        id: int
            refers to the order of discovery and storage (PointPile)
        children: list of nodes
            children nodes. if None, a empty list is used.

        """
        self.value = value
        self.id = id
        if children is None:
            self.children = []
        else:
            self.children = children

    def __str__(self, indent=0):
        """Visual representation of the node and its children (recursive)."""
        return '\n'.join([' ' * indent + '- Node: %s' % self.value] +
                         [c.__str__(indent=indent + 2) for c in self.children])

    def __lt__(self, other):
        """Define order of node based on value attribute."""
        return self.value < other.value


multitree_sort_key = operator.itemgetter(1)


class BreadthFirstIterator(object):
    """Generator exploring the tree.

    Nodes are ordered by value and expanded in order.
    The number of edges passing the node "in parallel" are "active".
    """

    def __init__(self, roots):
        """Start with initial set of nodes *roots*."""
        self.roots = roots
        self.reset()

    def reset(self):
        """(Re)start exploration from the top."""
        self.active_nodes = list(self.roots)
        self.active_root_ids = np.arange(len(self.active_nodes))
        self.active_node_values = np.array([n.value for n in self.active_nodes])
        self.active_node_ids = np.array([n.id for n in self.active_nodes])
        assert len(self.active_nodes) == len(self.active_root_ids)
        assert len(self.active_nodes) == len(self.active_node_values)
        # print("starting live points from %d roots" % len(self.roots), len(self.active_nodes))

    def next_node(self):
        """Get next node in order.

        Does not remove the node from active set.

        Returns
        --------
        None if done.
        rootid, node, (active_nodes, active_root_ids, active_node_values, active_node_ids)
        otherwise

        """
        if self.active_nodes == []:
            return None
        self.next_index = np.argmin(self.active_node_values)
        i = self.next_index
        node = self.active_nodes[i]
        rootid = self.active_root_ids[i]
        assert not isinstance(rootid, float)
        # print("consuming %.1f" % node.value, len(node.children), 'nlive:', len(self.active_nodes))
        assert len(self.active_nodes) == len(self.active_root_ids)
        assert len(self.active_nodes) == len(self.active_node_values)
        return rootid, node, (self.active_nodes, self.active_root_ids, self.active_node_values, self.active_node_ids)

    def drop_next_node(self):
        """Forget the current node."""
        i = self.next_index
        mask = np.ones(len(self.active_nodes), dtype=bool)
        mask[i] = False
        self.active_nodes.pop(i)
        self.active_node_values = self.active_node_values[mask]
        self.active_root_ids = self.active_root_ids[mask]
        self.active_node_ids = self.active_node_ids[mask]
        assert len(self.active_nodes) == len(self.active_root_ids)
        assert len(self.active_nodes) == len(self.active_node_values)

    def expand_children_of(self, rootid, node):
        """Replace the current node with its children.

        rootid and node have to come from the most recent call to next_node.
        """
        # print("replacing %.1f" % node.value, len(node.children))
        i = self.next_index
        newnnodes = len(self.active_nodes) - 1 + len(node.children)
        if len(node.children) == 1:
            self.active_nodes[i] = node.children[0]
            self.active_node_values[i] = node.children[0].value
            self.active_root_ids[i] = rootid
            self.active_node_ids[i] = node.children[0].id
        else:
            mask = np.ones(len(self.active_nodes), dtype=bool)
            mask[i] = False
            self.active_nodes.pop(i)
            if len(node.children) == 0:
                self.active_node_values = self.active_node_values[mask]
                self.active_root_ids = self.active_root_ids[mask]
                self.active_node_ids = self.active_node_ids[mask]
            else:
                self.active_nodes += node.children
                self.active_node_values = np.concatenate((self.active_node_values[mask], [c.value for c in node.children]))
                # print(self.active_root_ids, '+', [rootid for c in node.children], '-->')
                self.active_root_ids = np.concatenate((self.active_root_ids[mask], [rootid for c in node.children]))
                self.active_node_ids = np.concatenate((self.active_node_ids[mask], [c.id for c in node.children]))
                # print(self.active_root_ids)
            assert len(self.active_nodes) == len(self.active_root_ids)
            assert len(self.active_nodes) == len(self.active_node_values)
            assert len(self.active_nodes) == len(self.active_node_ids)
        assert newnnodes == len(self.active_nodes), (len(self.active_nodes), newnnodes, len(node.children))
        assert newnnodes == len(self.active_root_ids), (len(self.active_root_ids), newnnodes, len(node.children))
        assert newnnodes == len(self.active_node_values), (len(self.active_node_values), newnnodes, len(node.children))
        assert newnnodes == len(self.active_node_ids), (len(self.active_node_ids), newnnodes, len(node.children))


def print_tree(roots, title='Tree:'):
    """Print a pretty yet compact graphic of the tree."""
    print()
    print(title)
    explorer = BreadthFirstIterator(roots)
    lanes = list(roots)
    lastlane = -1

    while True:
        next_node = explorer.next_node()
        if next_node is None:
            break

        rootid, node, (active_nodes, active_rootids, active_values, active_nodeids) = next_node
        laneid = lanes.index(node)
        nchildren = len(node.children)
        leftstr = ''.join([' ' if n is None else '║' for n in lanes[:laneid]])
        rightstr = ''.join([' ' if n is None else '║' for n in lanes[laneid + 1:]])

        if lastlane == laneid:
            sys.stdout.write(leftstr + '║' + rightstr + "\n")
        rightstr = rightstr + " \t" + str(node.value)
        if nchildren == 0:
            sys.stdout.write(leftstr + 'O' + rightstr + "\n")
            lanes[laneid] = None  # keep lane empty
        elif nchildren == 1:
            sys.stdout.write(leftstr + '+' + rightstr + "\n")
            lanes[laneid] = node.children[0]
        else:
            # expand width:
            for j, child in enumerate(node.children):
                rightstr2 = ''.join([' ' if n is None else '\\' for n in lanes[laneid + 1:]])
                if len(rightstr2) != 0:
                    sys.stdout.write(leftstr + '║' + ' ' * j + rightstr2 + "\n")
            sys.stdout.write(leftstr + '╠' + '╦' * (nchildren - 2) + '╗' + rightstr + "\n")

            lanes.pop(laneid)
            for j, child in enumerate(node.children):
                lanes.insert(laneid, child)
        explorer.expand_children_of(rootid, node)
        lastlane = laneid


def dump_tree(filename, roots, pointpile):
    """Write a copy of the tree to a HDF5 file."""
    import h5py

    nodes_from_ids = []
    nodes_to_ids = []
    nodes_values = []

    explorer = BreadthFirstIterator(roots)
    while True:
        next_node = explorer.next_node()
        if next_node is None:
            break
        rootid, node, (active_nodes, active_rootids, active_values, active_nodeids) = next_node
        for c in node.children:
            nodes_from_ids.append(node.id)
            nodes_to_ids.append(c.id)
            nodes_values.append(c.value)
        explorer.expand_children_of(rootid, node)

    with h5py.File(filename, 'w') as f:
        f.create_dataset('unit_points', data=pointpile.us[:pointpile.nrows,:], compression='gzip', shuffle=True)
        f.create_dataset('points', data=pointpile.ps[:pointpile.nrows,:], compression='gzip', shuffle=True)

        f.create_dataset('nodes_parent_id', data=nodes_from_ids, compression='gzip', shuffle=True)
        f.create_dataset('nodes_child_id', data=nodes_to_ids, compression='gzip', shuffle=True)
        f.create_dataset('nodes_child_logl', data=nodes_values, compression='gzip', shuffle=True)


def count_tree(roots):
    """Return the maximum number of parallel edges and the total number of nodes."""
    explorer = BreadthFirstIterator(roots)
    nnodes = 0
    maxwidth = 0

    while True:
        next_node = explorer.next_node()
        if next_node is None:
            return nnodes, maxwidth
        rootid, node, (active_nodes, active_rootids, active_values, active_nodeids) = next_node
        maxwidth = max(maxwidth, len(active_rootids))
        nnodes += 1
        explorer.expand_children_of(rootid, node)


def count_tree_between(roots, lo, hi):
    """Build basic tree statistics.

    Returns
    --------
    nnodes: int
        the total number of nodes in the value interval lo .. hi (inclusive).
    maxwidth: int
        the maximum number of parallel edges

    """
    explorer = BreadthFirstIterator(roots)
    nnodes = 0
    maxwidth = 0

    while True:
        next_node = explorer.next_node()
        if next_node is None:
            return nnodes, maxwidth

        rootid, node, (active_nodes, active_rootids, active_values, active_nodeids) = next_node

        if node.value > hi:
            # can stop already
            return nnodes, maxwidth

        if lo <= node.value <= hi:
            maxwidth = max(maxwidth, len(active_rootids))
            nnodes += 1

        explorer.expand_children_of(rootid, node)


def find_nodes_before(root, value):
    """Identify all nodes that have children above value.

    If a root child is above the value, its parent (root) is the leaf returned.

    Returns
    --------
    list_of_parents: list of nodes
        parents
    list_of_nforks: list of floats
        The list of number of forks experienced is:
        1 if direct descendent of one of the root node's children,
        where no node had more than one child.
        12 if the root child had 4 children, one of which had 3 children.

    """
    roots = root.children
    parents = []
    parent_weights = []

    weights = {n.id: 1. for n in roots}
    explorer = BreadthFirstIterator(roots)
    while True:
        next_node = explorer.next_node()
        if next_node is None:
            break
        rootid, node, _ = next_node
        if node.value >= value:
            # already past (root child)
            parents.append(root)
            parent_weights.append(1)
            break
        elif any(n.value >= value for n in node.children):
            # found matching parent
            parents.append(node)
            parent_weights.append(weights[node.id])
            explorer.drop_next_node()
        else:
            # continue exploring
            explorer.expand_children_of(rootid, node)
            weights.update({n.id: weights[node.id] * len(node.children)
                for n in node.children})
        del weights[node.id]
    return parents, parent_weights


class PointPile(object):
    """A in-memory linearized storage of point coordinates.

    TreeNodes only store the logL value and id,
    which is the index in the point pile. The point pile stores
    the point coordinates.
    """

    def __init__(self, udim, pdim, chunksize=1000):
        """Set up point pile.

        Parameters
        -----------
        udim: int
            number of parameters, dimension of unit cube points
        pdim: int
            number of physical (and derived) parameters
        chunksize: int
            the point pile grows as needed, in these intervals.

        """
        self.nrows = 0
        self.chunksize = chunksize
        self.us = np.zeros((self.chunksize, udim))
        self.ps = np.zeros((self.chunksize, pdim))
        self.udim = udim
        self.pdim = pdim

    def add(self, newpointu, newpointp):
        """Save point `newpointu` (cube) / `newpointp` (parameters)."""
        if self.nrows >= self.us.shape[0]:
            self.us = np.concatenate((self.us, np.zeros((self.chunksize, self.udim))))
            self.ps = np.concatenate((self.ps, np.zeros((self.chunksize, self.pdim))))
        assert len(newpointu) == self.us.shape[1], (newpointu, self.us.shape)
        assert len(newpointp) == self.ps.shape[1], (newpointp, self.ps.shape)
        self.us[self.nrows,:] = newpointu
        self.ps[self.nrows,:] = newpointp
        self.nrows += 1
        return self.nrows - 1

    def getu(self, i):
        """Get cube point(s) with index(indices) `i`."""
        return self.us[i]

    def getp(self, i):
        """Get parameter point(s) with index(indices) `i`."""
        return self.ps[i]

    def make_node(self, value, u, p):
        """Save point `u` (cube) / `p` (parameters), return a tree node."""
        index = self.add(u, p)
        return TreeNode(value=value, id=index)


class SingleCounter(object):
    """Evidence log(Z) and posterior weight summation for a Nested Sampling tree."""

    def __init__(self, random=False):
        """Initialise counter.

        Parameters
        ----------
        random: bool
            if False, use mean estimator for volume shrinkage
            if True, draw a random sample

        """
        self.reset()
        self.random = random

    def reset(self):
        """Reset counters and integration."""
        self.logweights = []
        self.H = None
        self.logZ = -np.inf
        self.logZerr = np.inf
        self.logVolremaining = 0
        self.i = 0
        self.fraction_remaining = np.inf
        self.Lmax = -np.inf

    @property
    def logZremain(self):
        """Estimate conservatively the logZ of the current tail (un-opened nodes)."""
        return self.Lmax + self.logVolremaining

    def passing_node(self, node, parallel_nodes):
        """Accumulate node to the integration.

        Parameters
        -----------
        node: TreeNode
            breadth-first removed node
        parallel_nodes: list
            nodes active next to node

        """
        # node is being consumed
        # we have parallel arcs to parallel_nodes

        nchildren = len(node.children)
        Li = node.value
        nlive = len(parallel_nodes)

        if nchildren >= 1:
            # one arc terminates, another is spawned

            # weight is the size of the slice off the volume
            logleft = log1p(-exp(-1. / nlive))
            logright = -1. / nlive
            if self.random:
                randompoint = np.random.beta(1, nlive)
                logleft = log(randompoint)
                logright = log1p(-randompoint)

            logwidth = logleft + self.logVolremaining
            wi = logwidth + Li
            self.logweights.append(logwidth)
            if math.isinf(self.logZ):
                self.logZ = wi
                self.H = Li - self.logZ
            else:
                logZnew = logaddexp(self.logZ, wi)
                self.H = exp(wi - logZnew) * Li + exp(self.logZ - logZnew) * (self.H + self.logZ) - logZnew
                assert np.all(np.isfinite(self.H)), (self.H, wi, logZnew, Li, self.logZ)
                self.logZ = logZnew

            # print(self.H)
            # self.Lmax = max(node.value, self.Lmax)
            # self.Lmax = max((n.value for n in parallel_nodes))
            # logZremain = parallel_nodes.max() + self.logVolremaining
            # print("L=%.1f N=%d V=%.2e logw=%.2e logZ=%.1f logZremain=%.1f" % (Li, nlive, self.logVolremaining, wi, self.logZ, logZremain))
            # volume is reduced by exp(-1/N)
            self.logVolremaining += logright
            # TODO: this needs to change if nlive varies
            self.logZerr = (self.H / nlive)**0.5
            assert np.all(np.isfinite(self.logZerr)), (self.H, nlive)
        else:
            # contracting!

            # weight is simply volume / Nlive
            logwidth = self.logVolremaining - log(nlive)
            wi = logwidth + Li

            self.logweights.append(logwidth)
            self.logZ = logaddexp(self.logZ, wi)

            # print("L=%.1f N=%d V=%.2e logw=%.2e logZ=%.1f" % (Li, nlive, self.logVolremaining, wi, self.logZ))

            # the volume shrinks by (N - 1) / N
            # self.logVolremaining += log(1 - exp(-1. / nlive))
            # if nlive = 1, we are removing the last point, so remaining
            # volume is zero (leads to log of -inf, as expected)
            with np.errstate(divide='ignore'):
                self.logVolremaining += log1p(-1.0 / nlive)


class MultiCounter(object):
    """Like SingleCounter, but bootstrap capable.

    **Attributes**:
    
    - ``logZ``, ``logZerr``, ``logVolremaining``: main estimator
      ``logZerr`` is probably not reliable, because it needs ``nlive``
      to convert ``H`` to ``logZerr``.
    - ``Lmax``: highest loglikelihood currently known
    - ``logZ_bs``, ``logZerr_bs``: bootstrapped logZ estimate
    - ``logZremain``, ``remainder_ratio``: weight and fraction of the unexplored remainder

    Each of the following has as many entries as number of iterations:

    - ``all_H``, ``all_logZ``, ``all_logVolremaining``, ``logweights``:
      information for all instances
      first entry is the main estimator, i.e., not bootstrapped
    - ``istail``: whether that node was a leaf.
    - ``nlive``: number of parallel arcs ("live points")

    """

    def __init__(self, nroots, nbootstraps=10, random=False):
        """Initialise counter.

        Parameters
        ----------
        nroots: int
            number of children the tree root has
        nbootstraps: int
            number of bootstrap rounds
        random: bool
            if False, use mean estimator for volume shrinkage
            if True, draw a random sample

        """
        allyes = np.ones(nroots, dtype=bool)
        # the following is a masked array of size (nbootstraps+1, nroots)
        # which rootids are active in each bootstrap instance
        # the first one contains everything
        self.rootids = [allyes]
        # np.random.seed(1)
        for i in range(nbootstraps):
            mask = ~allyes
            rootids = np.unique(np.random.randint(nroots, size=nroots))
            mask[rootids] = True
            self.rootids.append(mask)
        self.rootids = np.array(self.rootids)
        self.random = random
        self.ncounters = len(self.rootids)

        self.reset(len(self.rootids))

    def reset(self, nentries):
        """Reset counters/integrator."""
        self.logweights = []
        self.istail = []
        self.logZ = -np.inf
        self.logZerr = np.inf
        self.all_H = -np.nan * np.ones(nentries)
        self.all_logZ = -np.inf * np.ones(nentries)
        self.all_logVolremaining = np.zeros(nentries)
        self.logVolremaining = 0.0
        self.Lmax = -np.inf

        self.all_logZremain = np.inf * np.ones(nentries)
        self.logZremainMax = np.inf
        self.logZremain = np.inf
        self.remainder_ratio = 1.0
        self.remainder_fraction = 1.0

    @property
    def logZ_bs(self):
        """Estimate logZ from the bootstrap ensemble."""
        return self.all_logZ[1:].mean()

    @property
    def logZerr_bs(self):
        """Estimate logZ error from the bootstrap ensemble."""
        return self.all_logZ[1:].std()

    def passing_node(self, rootid, node, rootids, parallel_values):
        """Accumulate node to the integration.

        Breadth-first removed `node` and nodes active next to node (`parallel_nodes`).
        rootid and rootids are needed to identify which bootstrap instance
        should accumulate.

        Parameters
        ----------
        rootid: TreeNode
            root node this `node` is from.
        node: TreeNode
            node being processed.
        rootids: array of ints
            for each parallel node, which root it belongs to.
        parallel_nodes: array of TreeNodes
            parallel nodes passing `node`.

        """
        # node is being consumed
        # we have parallel arcs to parallel_nodes

        assert not isinstance(rootid, float)
        nchildren = len(node.children)
        Li = node.value
        # in which bootstraps is rootid?
        active = self.rootids[:,rootid]
        # how many live points does each bootstrap have?
        nlive = self.rootids[:,rootids].sum(axis=1)
        nlive0 = nlive[0]

        if nchildren >= 1:
            # one arc terminates, another is spawned

            # weight is the size of the slice off the volume
            if self.random:
                randompoint = np.random.beta(1, nlive, size=self.ncounters)
                logleft = log(randompoint)
                logright = log1p(-randompoint)
                logleft[0] = log1p(-exp(-1. / nlive0))
                logright[0] = -1. / nlive0
            else:
                logleft = log1p(-exp(-1. / nlive))
                logright = -1. / nlive

            logwidth = logleft + self.all_logVolremaining
            logwidth[~active] = -np.inf
            wi = logwidth[active] + Li
            self.logweights.append(logwidth)
            self.istail.append(False)

            # print("updating continuation...", Li)
            assert active[0], (active, rootid)
            logZ = self.all_logZ[active]
            logZnew = logaddexp(logZ, wi)
            H = exp(wi - logZnew) * Li + exp(logZ - logZnew) * (self.all_H[active] + logZ) - logZnew
            first_setting = np.isinf(logZ)
            # print()
            # print("Hnext:", H[0], first_setting[0])
            assert np.isfinite(H[~first_setting]).all(), (first_setting, self.all_H[active][~first_setting], H, wi, logZnew, Li, logZ)
            self.all_logZ[active] = np.where(first_setting, wi, logZnew)
            # print("logZ:", self.all_logZ[0])
            if first_setting[0]:
                assert np.all(np.isfinite(Li - wi)), (Li, wi)
            else:
                assert np.isfinite(self.all_H[0]), self.all_H[0]
                assert np.isfinite(H[0]), (first_setting[0], H[0], self.all_H[0], wi[0], logZnew[0], Li, logZ[0])
            self.all_H[active] = np.where(first_setting, -logwidth[active], H)
            # print("H:", self.all_H[0])
            assert np.isfinite(self.all_H[active]).all(), (self.all_H[active], first_setting[0], H[0], self.all_H[0], wi[0], logZnew[0], Li, logZ[0])
            # assert np.all(np.isfinite(self.all_H[active])), (H, self.all_H[active], wi, logZnew, Li, logZ)
            self.logZ = self.all_logZ[0]

            # self.Lmax = max((n.value for n in parallel_nodes))
            # print("L=%.1f N=%d V=%.2e logw=%.2e logZ=%.1f logZremain=%.1f" % (Li, nlive[0], self.logVolremaining, wi[0], self.logZ, logZremain))
            # print("L=%.1f N=%d V=%.2e logw=%.2e logZ=%.1f logZremain=%.1f" % (Li, nlive[0], self.all_logVolremaining[0], (logwidth + Li)[0], self.all_logZ[0], logZremain))
            # print("L=%.1f N=%d V=%.2e logw=%.2e logZ=<%.1f logZremain=%.1f" % (Li, nlive[1], self.all_logVolremaining[1], (logwidth + Li)[1], self.all_logZ[1], logZremain))

            if self.all_H[0] > 0:
                # TODO: this needs to change if nlive varies
                self.logZerr = (self.all_H[0] / nlive0)**0.5
            # assert np.all(np.isfinite(self.logZerr)), (self.logZerr, self.all_H[0], nlive)

            # volume is reduced by exp(-1/N)
            self.all_logVolremaining[active] += logright[active]
            self.logVolremaining = self.all_logVolremaining[0]
        else:
            # contracting!
            # print("contracting...", Li)

            # weight is simply volume / Nlive
            logwidth = -np.inf * np.ones(self.ncounters)
            logwidth[active] = self.all_logVolremaining[active] - log(nlive[active])
            wi = logwidth + Li

            self.logweights.append(logwidth)
            self.istail.append(True)
            self.all_logZ[active] = logaddexp(self.all_logZ[active], wi[active])
            self.logZ = self.all_logZ[0]

            # print("L=%.1f N=%d V=%.2e logw=%.2e logZ=%.1f" % (Li, nlive, self.logVolremaining, wi, self.logZ))

            # the volume shrinks by (N - 1) / N
            # self.logVolremaining += log(1 - exp(-1. / nlive))
            # if nlive = 1, we are removing the last point, so remaining
            # volume is zero (leads to log of -inf, as expected)
            with np.errstate(divide='ignore'):
                self.all_logVolremaining[active] += log1p(-1.0 / nlive[active])
            self.logVolremaining = self.all_logVolremaining[0]

        V = self.all_logVolremaining - log(nlive0)
        Lmax = np.max(parallel_values)
        self.all_logZremain = V + log(np.sum(exp(parallel_values - Lmax))) + Lmax
        self.logZremainMax = self.all_logZremain.max()
        self.logZremain = self.all_logZremain[0]
        with np.errstate(over='ignore', under='ignore'):
            self.remainder_ratio = exp(self.logZremain - self.logZ)
            self.remainder_fraction = 1.0 / (1 + exp(self.logZ - self.logZremain))


def logz_sequence(root, pointpile, nbootstraps=12):
    """Run MultiCounter through tree `root`.

    Keeps track of, and returns ``(logz, logzerr, logv, nlive)``.
    """
    roots = root.children

    explorer = BreadthFirstIterator(roots)
    main_iterator = MultiCounter(nroots=len(roots), nbootstraps=nbootstraps, random=True)
    main_iterator.Lmax = max(n.value for n in roots)
    logz = []
    logzerr = []
    nlive = []
    nodeids = []
    logvol = []
    logl = []
    niter = 0

    while True:
        next_node = explorer.next_node()
        if next_node is None:
            break
        rootid, node, (active_nodes, active_rootids, active_values, active_node_ids) = next_node

        main_iterator.passing_node(rootid, node, active_rootids, active_values)

        logz.append(main_iterator.logZ)
        with np.errstate(invalid='ignore'):
            # first time they are all the same
            logzerr.append(main_iterator.logZerr_bs)
        nlive.append(len(active_values))
        logvol.append(main_iterator.logVolremaining)
        nodeids.append(node.id)
        logl.append(node.value)

        if node.children:
            main_iterator.Lmax = max(main_iterator.Lmax, max(n.value for n in node.children))
        explorer.expand_children_of(rootid, node)
        niter += 1

    logwt = np.asarray(logl) + np.asarray(main_iterator.logweights)[:,0]
    logvol[-1] = logvol[-2]

    return dict(logz=np.asarray(logz),
        logzerr=np.asarray(logzerr),
        logvol=np.asarray(logvol),
        samples_n=np.asarray(nlive),
        logwt=logwt,
        niter=niter,
        logl=np.asarray(logl),
        samples=pointpile.getp(nodeids),
    )
