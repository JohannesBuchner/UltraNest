#!/bin/bash
# Diagnostics campaign: 6 problems x d=4, 10, 40, 100.
# Sequential campaigns (one heavy task at a time), 5 joblib workers each.
# Replicates per run: 80 at d < 20, 8 at 20 <= d < 40, 4 at d >= 40.
# BLAS threads pinned to 1: the forked pool workers must not share BLAS
# mutexes (fork deadlock with matmul).
PY=/home/user/miniforge3/envs/UltraNest-fix/bin/python
cd /home/user/Downloads/tester/UltraNest-edit
for dim in 4 10 40 100; do
    nrep=80
    if [ $dim -ge 40 ]; then nrep=4
    elif [ $dim -ge 20 ]; then nrep=8
    fi
    for prob in gauss funnel banana hyperpyramid shell splitellipsoid; do
        echo "=== $prob d=$dim nrep=$nrep start $(date) ==="
        $PY evaluate/slice_diagnostics.py --problem $prob --dim $dim \
            --nrep $nrep --workers 5 || echo "FAILED $prob d=$dim"
        echo "=== $prob d=$dim done $(date) ==="
    done
done
echo "ALL CAMPAIGNS DONE $(date)"
