# Makefile for the perfectns testing pipeline
#
# Each (problem, dimension) x (integrator, proposal) combination writes
# its plots and results into
#
#     output/<problemname>-<dim>/<integrator>-<proposal>/
#
# and its final output file is results.json in that folder.
#
# Usage:
#     make                          # build the default matrix
#     make output/banana-10d/snowballingns-quadratic-slice/results.json
#     make PROBLEMS=funnel DIMS=10 COMBOS="snowballingns-quadratic-slice"
#     make K=100 MAX_ROTATIONS=3 output/banana-2d/snowballingns-mix/results.json
#     make clean

PROBLEMS ?= banana gauss
DIMS ?= 2 10
COMBOS ?= ultranest-mlfriends ultranest-de-mix \
          snowballingns-quadratic-slice snowballingns-mix

NLIVE ?= 400
K ?= 400
P ?= 20
NSTEPS ?= 1
FRAC_REMAIN ?= 0.5
MAX_ROTATIONS ?= 8
SEED ?= 1

PYTHON ?= python
PIPELINE := pipeline.py

COMBINATIONS := $(foreach p,$(PROBLEMS),$(foreach d,$(DIMS),$(foreach c,$(COMBOS),$(p)-$(d)d/$(c))))
TARGETS := $(addprefix output/,$(addsuffix /results.json,$(COMBINATIONS)))

.PHONY: all clean
.DEFAULT_GOAL := all
all: $(TARGETS)

# one rule per combination: running the pipeline produces the results
# file and, alongside it, the plots
define COMBO_RULE
output/$(1)-$(2)d/$(3)/results.json: $(PIPELINE)
	$(PYTHON) $(PIPELINE) --problem $(1) --dim $(2) --combo $(3) --outdir output \
		--nlive $(NLIVE) --K $(K) --P $(P) --nsteps $(NSTEPS) \
		--frac-remain $(FRAC_REMAIN) --max-rotations $(MAX_ROTATIONS) \
		--seed $(SEED)
endef
$(foreach p,$(PROBLEMS),$(foreach d,$(DIMS),$(foreach c,$(COMBOS),$(eval $(call COMBO_RULE,$(p),$(d),$(c))))))

clean:
	rm -rf output
