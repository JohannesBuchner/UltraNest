PYTHON := PYTHONPATH=.. python3

all: circgauss asymgauss pyramid multigauss shell

evaluate_sampling_asymgauss_2d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=asymgauss --x_dim=2 --nsteps=2500

evaluate_sampling_asymgauss_10d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=asymgauss --x_dim=10 --nsteps=10000

evaluate_sampling_asymgauss_100d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=asymgauss --x_dim=100 --nsteps=5000

asymgauss: evaluate_sampling_asymgauss_2d_N400_shrinkage.pdf evaluate_sampling_asymgauss_10d_N400_shrinkage.pdf

evaluate_sampling_circgauss_2d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=circgauss --x_dim=2 --nsteps=2500

evaluate_sampling_circgauss_10d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=circgauss --x_dim=10 --nsteps=10000

circgauss: evaluate_sampling_circgauss_2d_N400_shrinkage.pdf evaluate_sampling_circgauss_10d_N400_shrinkage.pdf

evaluate_sampling_pyramid_2d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=pyramid --x_dim=2 --nsteps=2500

evaluate_sampling_pyramid_10d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=pyramid --x_dim=10 --nsteps=10000

evaluate_sampling_pyramid_100d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=pyramid --x_dim=100 --nsteps=5000

pyramid: evaluate_sampling_pyramid_2d_N400_shrinkage.pdf evaluate_sampling_pyramid_10d_N400_shrinkage.pdf

evaluate_sampling_multigauss_2d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=multigauss --x_dim=2 --nsteps=2500

evaluate_sampling_multigauss_10d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=multigauss --x_dim=10 --nsteps=10000

multigauss: evaluate_sampling_multigauss_2d_N400_shrinkage.pdf evaluate_sampling_multigauss_10d_N400_shrinkage.pdf

evaluate_sampling_shell_2d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=shell --x_dim=2 --nsteps=3000

evaluate_sampling_shell_10d_N400_shrinkage.pdf: evaluate_sampling.py
	${PYTHON} evaluate_sampling.py --problem=shell --x_dim=10 --nsteps=3300

shell: evaluate_sampling_shell_2d_N400_shrinkage.pdf evaluate_sampling_shell_10d_N400_shrinkage.pdf
