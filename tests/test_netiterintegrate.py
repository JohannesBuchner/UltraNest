from __future__ import print_function, division
import os
import numpy as np
from ultranest.store import TextPointStore
from ultranest.netiter import PointPile, TreeNode, count_tree, print_tree, dump_tree
from ultranest.netiter import SingleCounter, MultiCounter, BreadthFirstIterator



def integrate_singleblock(num_live_points, pointstore, x_dim, num_params, dlogz=0.5):
	active_u = []
	active_v = []
	active_logl = []
	for i in range(num_live_points):
		idx, row = pointstore.pop(-np.inf)
		assert row is not None
		active_u.append(row[2:2+x_dim])
		active_v.append(row[2+x_dim:2+x_dim+num_params])
		active_logl.append(row[1])

	saved_v = []  # Stored points for posterior results
	saved_logl = []
	saved_logwt = []
	h = 0.0  # Information, initially 0.
	logz = -1e300  # ln(Evidence Z), initially Z=0
	logvol = 0
	logvolf = np.log1p(- np.exp(-1.0 / num_live_points))
	#fraction_remain = 1.0
	max_iters = 10000000

	for it in range(0, max_iters):

		# Worst object in collection and its weight (= volume * likelihood)
		worst = np.argmin(active_logl)
		logwt = logvol + logvolf + active_logl[worst]

		# Update evidence Z and information h.
		logz_new = np.logaddexp(logz, logwt)
		h = (np.exp(logwt - logz_new) * active_logl[worst] + np.exp(logz - logz_new) * (h + logz) - logz_new)
		logz = logz_new
		logz_remain = np.max(active_logl) - it / num_live_points

		#print("L=%.1f N=%d V=%.2e logw=%.2e logZ=%.1f logZremain=%.1f" % (active_logl[worst], num_live_points, logvol, logwt, logz, logz_remain))
		# Shrink interval
		logvol -= 1.0 / num_live_points

		# Add worst object to samples.
		saved_v.append(np.array(active_v[worst]))
		saved_logwt.append(logwt)
		saved_logl.append(active_logl[worst])

		# The new likelihood constraint is that of the worst object.
		loglstar = active_logl[worst]

		idx, row = pointstore.pop(loglstar)
		assert row is not None
		u = row[2:2+x_dim]
		v = row[2+x_dim:2+x_dim+num_params]
		logl = row[1]
		
		active_u[worst] = u
		active_v[worst] = v
		active_logl[worst] = logl

		#fraction_remain = np.logaddexp(logz, logz_remain) - logz

		# Stopping criterion
		if logz_remain < logz:
			break
	
	logvol = -len(saved_v) / num_live_points
	for i in np.argsort(active_logl):
		logwt = logvol - np.log(num_live_points) + active_logl[i]
		logz_new = np.logaddexp(logz, logwt)
		h = (np.exp(logwt - logz_new) * active_logl[i] + np.exp(logz - logz_new) * (h + logz) - logz_new)
		logz = logz_new
		#print("L=%.1f N=%d V=%.2e logw=%.2e logZ=%.1f" % (active_logl[i], num_live_points, logvol, logwt, logz))
		saved_v.append(np.array(active_v[i]))
		saved_logwt.append(logwt)
		saved_logl.append(active_logl[i])

	saved_v = np.array(saved_v)
	saved_wt = np.exp(np.array(saved_logwt) - logz)
	saved_logl = np.array(saved_logl)
	logzerr = np.sqrt(h / num_live_points)

	results = dict(niter=it, logz=logz, logzerr=logzerr,
		weighted_samples=dict(v=saved_v, w = saved_wt, logw = saved_logwt, L=saved_logl),
	)
	
	return results


def strategy_advice(node, parallel_values, main_iterator, counting_iterators, rootid):
	if len(node.children) > 0:
		# we don't expand if node already has children
		print("not expanding, already has children")
		assert False
		return np.nan, np.nan
	
	Lmin = parallel_values.min()
	Lmax = parallel_values.max()
	
	logZremain = main_iterator.logZremain
	
	# if the remainder dominates, return that range
	if logZremain > main_iterator.logZ:
		return Lmin, Lmax
	
	#print("not expanding, remainder not dominant")
	return np.nan, np.nan

class __Point(object):
	def __init__(self, u, p):
		self.u = u
		self.p = p


def integrate_graph_singleblock(num_live_points, pointstore, x_dim, num_params, dlogz=0.5):
	pp = PointPile(x_dim, num_params)
	def create_node(pointstore, Lmin):
		idx, row = pointstore.pop(Lmin)
		assert row is not None
		L = row[1]
		u = row[2:2+x_dim]
		p = row[2+x_dim:2+x_dim+num_params]
		assert np.isfinite(L)
		return pp.make_node(L, u, p)
	
	# we create a bunch of live points from the prior volume
	# each of which is the start of a chord (in the simplest case)
	roots = [create_node(pointstore, -np.inf) for i in range(num_live_points)]
	
	iterator_roots = []
	np.random.seed(1)
	for i in range(10):
		# boot-strap which roots are assigned to this iterator
		rootids = np.unique(np.random.randint(len(roots), size=len(roots)))
		#print(rootids)
		iterator_roots.append((SingleCounter(random=True), rootids))
	
	# and we have one that operators on the entire tree
	main_iterator = SingleCounter()
	main_iterator.Lmax = max(n.value for n in roots)
	assert np.isfinite(main_iterator.Lmax)
	
	explorer = BreadthFirstIterator(roots)
	Llo, Lhi = -np.inf, np.inf
	strategy_stale = True
	
	saved_nodeids = []
	saved_logl = []
	
	# we go through each live point (regardless of root) by likelihood value
	while True:
		#print()
		next = explorer.next_node()
		if next is None:
			break
		rootid, node, (active_nodes, active_rootids, active_values, active_node_ids) = next
		# this is the likelihood level we have to improve upon
		Lmin = node.value
		
		saved_nodeids.append(node.id)
		saved_logl.append(Lmin)
		
		expand_node = Lmin <= Lhi and Llo <= Lhi
		# if within suggested range, expand
		if strategy_stale or not (Lmin <= Lhi):
			# check with advisor if we want to expand this node
			Llo, Lhi = strategy_advice(node, active_values, main_iterator, [], rootid)
			#print("L range to expand:", Llo, Lhi, "have:", Lmin, "=>", Lmin <= Lhi, Llo <= Lhi)
			strategy_stale = False
		strategy_stale = True
		
		if expand_node:
			# sample a new point above Lmin
			#print("replacing node", Lmin, "from", rootid, "with", L)
			node.children.append(create_node(pointstore, Lmin))
			main_iterator.Lmax = max(main_iterator.Lmax, node.children[0].value)
		else:
			#print("ending node", Lmin)
			pass
		
		# inform iterators (if it is their business) about the arc
		main_iterator.passing_node(node, active_values)
		for it, rootids in iterator_roots:
			if rootid in rootids:
				mask = np.in1d(active_rootids, rootids, assume_unique=True)
				#mask1 = np.array([rootid2 in rootids for rootid2 in active_rootids])
				#assert (mask1 == mask).all(), (mask1, mask)
				it.passing_node(node, active_values[mask])
		#print([it.H for it,_ in iterator_roots])
		
		explorer.expand_children_of(rootid, node)
		
	# points with weights
	#saved_u = np.array([pp[nodeid].u for nodeid in saved_nodeids])
	saved_v = pp.getp(saved_nodeids)
	saved_logwt = np.array(main_iterator.logweights)
	saved_wt = np.exp(saved_logwt - main_iterator.logZ)
	saved_logl = np.array(saved_logl)
	print('%.4f +- %.4f (main)' % (main_iterator.logZ, main_iterator.logZerr))
	Zest = np.array([it.logZ for it, _ in iterator_roots])
	print('%.4f +- %.4f (bs)' % (Zest.mean(), Zest.std()))

	results = dict(niter=len(saved_logwt), 
		logz=main_iterator.logZ, logzerr=main_iterator.logZerr,
		weighted_samples=dict(v=saved_v, w = saved_wt, logw = saved_logwt, L=saved_logl),
		tree=TreeNode(-np.inf, children=roots),
	)
	
	# return entire tree
	return results


def multi_integrate_graph_singleblock(num_live_points, pointstore, x_dim, num_params, dlogz=0.5, withtests=False):
	pp = PointPile(x_dim, num_params)
	def create_node(pointstore, Lmin):
		idx, row = pointstore.pop(Lmin)
		assert row is not None
		L = row[1]
		u = row[2:2+x_dim]
		p = row[2+x_dim:2+x_dim+num_params]
		return pp.make_node(L, u, p)
	
	# we create a bunch of live points from the prior volume
	# each of which is the start of a chord (in the simplest case)
	roots = [create_node(pointstore, -np.inf) for i in range(num_live_points)]
	
	# and we have one that operators on the entire tree
	main_iterator = MultiCounter(nroots=len(roots), nbootstraps=10, random=True, check_insertion_order=withtests)
	main_iterator.Lmax = max(n.value for n in roots)
	
	explorer = BreadthFirstIterator(roots)
	Llo, Lhi = -np.inf, np.inf
	strategy_stale = True
	
	saved_nodeids = []
	saved_logl = []
	
	# we go through each live point (regardless of root) by likelihood value
	while True:
		#print()
		next = explorer.next_node()
		if next is None:
			break
		rootid, node, (active_nodes, active_rootids, active_values, active_nodeids) = next
		assert not isinstance(rootid, float)
		# this is the likelihood level we have to improve upon
		Lmin = node.value
		
		saved_nodeids.append(node.id)
		saved_logl.append(Lmin)
		
		expand_node = Lmin <= Lhi and Llo <= Lhi
		# if within suggested range, expand
		if strategy_stale or not (Lmin <= Lhi):
			# check with advisor if we want to expand this node
			Llo, Lhi = strategy_advice(node, active_values, main_iterator, [], rootid)
			#print("L range to expand:", Llo, Lhi, "have:", Lmin, "=>", Lmin <= Lhi, Llo <= Lhi)
			strategy_stale = False
		strategy_stale = True
		
		if expand_node:
			# sample a new point above Lmin
			node.children.append(create_node(pointstore, Lmin))
			main_iterator.Lmax = max(main_iterator.Lmax, node.children[0].value)
		else:
			#print("ending node", Lmin)
			pass
		
		# inform iterators (if it is their business) about the arc
		
		assert not isinstance(rootid, float)
		main_iterator.passing_node(rootid, node, active_rootids, active_values)
		
		explorer.expand_children_of(rootid, node)

	print('tree size:', count_tree(roots))
		
	# points with weights
	#saved_u = pp.getu(saved_nodeids)
	saved_v = pp.getp(saved_nodeids)
	saved_logwt = np.array(main_iterator.logweights)
	saved_wt = np.exp(saved_logwt - main_iterator.logZ)
	saved_logl = np.array(saved_logl)
	print('%.4f +- %.4f (main)' % (main_iterator.logZ, main_iterator.logZerr))
	print('%.4f +- %.4f (bs)' % (main_iterator.all_logZ[1:].mean(), main_iterator.all_logZ[1:].std()))
	if withtests:
		print("insertion order:", float(main_iterator.insertion_order_runlength))

	results = dict(niter=len(saved_logwt), 
		logz=main_iterator.logZ, logzerr=main_iterator.logZerr,
		weighted_samples=dict(v=saved_v, w = saved_wt, logw = saved_logwt, L=saved_logl),
		tree=TreeNode(-np.inf, children=roots),
	)
	
	# return entire tree
	return results

testfile = os.path.join(os.path.dirname(__file__), 'eggboxpoints.tsv')

import time
import pytest


@pytest.mark.parametrize("nlive", [100])
def test_singleblock(nlive):
	assert os.path.exists(testfile), ("%s does not exist" % testfile)
	print("="*80)
	print("NLIVE=%d " % nlive)
	print("Standard integrator")
	pointstore = TextPointStore(testfile, 2 + 2 + 2)
	t = time.time()
	result = integrate_singleblock(num_live_points=nlive, pointstore=pointstore, num_params=2, x_dim=2)
	print('  %(logz).1f +- %(logzerr).1f in %(niter)d iter' % result, '%.2fs' % (time.time() - t))
	pointstore.close()
	
	print("Graph integrator")
	pointstore = TextPointStore(testfile, 2 + 2 + 2)
	t = time.time()
	result2 = integrate_graph_singleblock(num_live_points=nlive, pointstore=pointstore, num_params=2, x_dim=2)
	print('  %(logz).1f +- %(logzerr).1f in %(niter)d iter' % result2, '%.2fs' % (time.time() - t))
	pointstore.close()
	assert np.isclose(result2['logz'], result['logz'])
	
	print("Vectorized graph integrator")
	pointstore = TextPointStore(testfile, 2 + 2 + 2)
	t = time.time()
	result3 = multi_integrate_graph_singleblock(num_live_points=nlive, pointstore=pointstore, num_params=2, x_dim=2)
	print('  %(logz).1f +- %(logzerr).1f in %(niter)d iter' % result3, '%.2fs' % (time.time() - t))
	pointstore.close()
	assert np.isclose(result3['logz'], result['logz'])

	print("Vectorized graph integrator with insertion order test")
	pointstore = TextPointStore(testfile, 2 + 2 + 2)
	t = time.time()
	result3 = multi_integrate_graph_singleblock(num_live_points=nlive, pointstore=pointstore, num_params=2, x_dim=2, withtests=True)
	print('  %(logz).1f +- %(logzerr).1f in %(niter)d iter' % result3, '%.2fs' % (time.time() - t))
	pointstore.close()
	assert np.isclose(result3['logz'], result['logz'])

def test_visualisation():
	print("testing tree visualisation...")
	pp = PointPile(1, 1)
	tree = TreeNode()
	for i in range(5):
		j = np.random.randint(1000)
		node = pp.make_node(j, np.array([j]), np.array([j]))
		for k in range(i):
			j = np.random.randint(1000)
			node2 = pp.make_node(j, [j], [j])
			node.children.append(node2)
		tree.children.append(node)
	print(tree)
	print_tree(tree.children, title='Empty Tree')
	

def test_treedump():
	print("testing tree dumping...")
	pp = PointPile(1, 1)
	tree = TreeNode()
	for i in range(5):
		j = np.random.randint(1000)
		node = pp.make_node(j, np.array([j]), np.array([j]))
		for k in range(i):
			j = np.random.randint(1000)
			node2 = pp.make_node(j, [j], [j])
			node.children.append(node2)
		tree.children.append(node)
	dump_tree("test_tree.hdf5", tree.children, pp)
	os.remove("test_tree.hdf5")
	dump_tree("test_tree.hdf5", roots=tree.children, pointpile=pp)
	dump_tree("test_tree.hdf5", tree.children, pp)
	os.remove("test_tree.hdf5")
	

if __name__ == '__main__':
	for nlive in [100, 400, 2000]:
		test_singleblock(nlive)
	#pointstore = TextPointStore(testfile, 2 + 2 + 2)
	#nlive = 400
	#multi_integrate_graph_singleblock(num_live_points=nlive, pointstore=pointstore, num_params=2, x_dim=2)
	#pointstore.close()
