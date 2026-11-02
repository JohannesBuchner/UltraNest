# Makefile for the perfectns testing pipeline
#
# Each (problem, dimension) x (integrator, proposal) combination writes
# its plots and results into
#
#     output/<problemname>-<dim>/<integrator>-<proposal>-K<K>/
#
# and its final output file is results.json in that folder.
#
# Usage:
#     make                          # build the default matrix
#     make output/banana-8d/snowballingns-quadratic-slice-seed1-K400/results.json
#     make PROBLEMS=funnel DIMS=8 COMBOS="snowballingns-quadratic-slice-seed1"
#     make K=100 MAX_ROTATIONS=3 output/banana-2d/snowballingns-mix-seed1-K100/results.json
#     make clean

PROBLEMS ?= banana gauss box funnel hyperpyramid
DIMS ?= 2 8
COMBOS ?= ultranest-mlfriends ultranest-de-mix \
          snowballingns-quadratic-slice-seed1 snowballingns-quadratic-slice-seed2 \
          snowballingns-quadratic-slice-seed3 snowballingns-quadratic-slice-seed4 \
          snowballingns-mix-seed1 snowballingns-mix-seed2 \
          snowballingns-mix-seed3 snowballingns-mix-seed4 \
          snowballingns-custom-quadratic-slice-seed1 snowballingns-custom-quadratic-slice-seed2 \
          snowballingns-custom-quadratic-slice-seed3 snowballingns-custom-quadratic-slice-seed4 \
          snowballingns-custom-mix-seed1 snowballingns-custom-mix-seed2 \
          snowballingns-custom-mix-seed3 snowballingns-custom-mix-seed4

K ?= 50
NSTEPS ?= 1
FRAC_REMAIN ?= 0.5
MAX_ROTATIONS ?= 20
SEED ?= 1

# the in-place built cython extensions match this interpreter; the
# default `python` (the ml env) cannot import them
PYTHON ?= /home/user/miniforge3/envs/UltraNest-fix/bin/python
PIPELINE := pipeline.py

COMBINATIONS := $(foreach p,$(PROBLEMS),$(foreach d,$(DIMS),$(foreach c,$(COMBOS),$(p)-$(d)d/$(c)-K$(K))))
TARGETS := $(addprefix output/,$(addsuffix /results.json,$(COMBINATIONS)))

.PHONY: all clean
.DEFAULT_GOAL := all
all: $(TARGETS)

# one rule per combination: running the pipeline produces the results
# file and, alongside it, the plots
define COMBO_RULE
output/$(1)-$(2)d/$(3)-K$(K)/results.json: $(PIPELINE)
	$(PYTHON) $(PIPELINE) --problem $(1) --dim $(2) --combo $(3) --outdir output \
		--K $(K) --nsteps $(NSTEPS) \
		--frac-remain $(FRAC_REMAIN) --max-rotations $(MAX_ROTATIONS) \
		--seed $(SEED)
endef
$(foreach p,$(PROBLEMS),$(foreach d,$(DIMS),$(foreach c,$(COMBOS),$(eval $(call COMBO_RULE,$(p),$(d),$(c))))))

clean:
	rm -rf output

# speed benchmark of the sampling hot path (gauss-50d and banana-20d,
# K=10000, first rotation capped at 10000 batch advances / 30000 dead
# points); the side-by-side matrix: the quadratic slice and mixture
# proposals on the mlfriends, covariance and cube regions; results
# into benchmarks/benchmark_speed-*.json
.PHONY: benchmark
benchmark:
	$(PYTHON) benchmark_speed.py --problem gauss --dim 50 \
		--P 100 --legacy-direction --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem gauss --dim 50 \
		--P 100 --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem gauss --dim 50 \
		--P 100 --region covariance --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem gauss --dim 50 \
		--P 100 --region cube --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem gauss --dim 50 \
		--P 100 --proposal mix --region mlfriends --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem gauss --dim 50 \
		--P 100 --proposal mix --region covariance --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem gauss --dim 50 \
		--P 100 --proposal mix --region cube --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem banana --dim 20 \
		--P 100 --legacy-direction --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem banana --dim 20 \
		--P 100 --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem banana --dim 20 \
		--P 100 --region covariance --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem banana --dim 20 \
		--P 100 --region cube --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem banana --dim 20 \
		--P 100 --proposal mix --region mlfriends --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem banana --dim 20 \
		--P 100 --proposal mix --region covariance --outdir benchmarks
	$(PYTHON) benchmark_speed.py --problem banana --dim 20 \
		--P 100 --proposal mix --region cube --outdir benchmarks
