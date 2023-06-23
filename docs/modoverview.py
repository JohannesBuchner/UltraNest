import importlib

sections = [
	('Modules commonly used directly', ['integrator', 'hotstart', 'plot', 'stepsampler', 'popstepsampler', 'solvecompat']),
	('Internally used modules', ['mlfriends', 'netiter', 'ordertest', 'stepfuncs', 'store', 'viz']),
	('Experimental modules, no guarantees', ['dychmc', 'dyhmc', 'flatnuts', 'pathsampler', 'samplingpath']),
]

fout = open('API.rst', 'w')
fout.write("""API
===

`Full API documentation on one page <ultranest.html>`_

The main interface is :py:class:`ultranest.integrator.ReactiveNestedSampler`, 
also available as `ultranest.ReactiveNestedSampler`.

""")

for section, modules in sections:
	fout.write("\n%s:\n%s\n\n" % (section, '-'*80))
	for mod in modules:
		moddoc = importlib.import_module('ultranest.%s' % mod).__doc__
		modtitle = moddoc.strip().split('\n')[0]
		
		print('%-15s: %s' % (mod, modtitle))
		fout.write(" * :py:mod:`ultranest.%s`: %s\n" % (mod, modtitle))

fout.write("""

Alphabetical list of submodules
-------------------------------

.. toctree::
   :maxdepth: 2

   ultranest


""")
