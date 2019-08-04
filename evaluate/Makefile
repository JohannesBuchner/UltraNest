PYTHON := python3

all: circgauss asymgauss pyramid multigauss shell

evaluate_sampling_asymgauss_2d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=asymgauss --x_dim=2 --nsteps=2500

evaluate_sampling_asymgauss_10d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=asymgauss --x_dim=10 --nsteps=10000

asymgauss: evaluate_sampling_asymgauss_2d_N200_shrinkage.pdf evaluate_sampling_asymgauss_10d_N200_shrinkage.pdf

evaluate_sampling_circgauss_2d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=circgauss --x_dim=2 --nsteps=2500

evaluate_sampling_circgauss_10d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=circgauss --x_dim=10 --nsteps=10000

circgauss: evaluate_sampling_circgauss_2d_N200_shrinkage.pdf evaluate_sampling_circgauss_10d_N200_shrinkage.pdf

evaluate_sampling_pyramid_2d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=pyramid --x_dim=2 --nsteps=2500

evaluate_sampling_pyramid_10d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=pyramid --x_dim=10 --nsteps=10000

pyramid: evaluate_sampling_pyramid_2d_N200_shrinkage.pdf evaluate_sampling_pyramid_10d_N200_shrinkage.pdf

evaluate_sampling_multigauss_2d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=multigauss --x_dim=2 --nsteps=2500

evaluate_sampling_multigauss_10d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=multigauss --x_dim=10 --nsteps=10000

multigauss: evaluate_sampling_multigauss_2d_N200_shrinkage.pdf evaluate_sampling_multigauss_10d_N200_shrinkage.pdf

evaluate_sampling_shell_2d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=shell --x_dim=2 --nsteps=2500

evaluate_sampling_shell_10d_N200_shrinkage.pdf:
	${PYTHON} evaluate_sampling.py --problem=shell --x_dim=10 --nsteps=2500

shell: evaluate_sampling_shell_2d_N200_shrinkage.pdf evaluate_sampling_shell_10d_N200_shrinkage.pdf





