python -u /home/pbloem/git/gated-rgcn/experiments/lp.bias.py \
        -m distmult -E 256 -e 400 --eval-int 50 --opt adagrad -B 1024 -N 500 0 500 --test-batch 100 -l 0.1 --loss ce \
        --reg-exp 3 --reg-eweight 3e-12 --reg-rweight 8e-15 --edropout 0.4 --rdropout 0.4 \
        --init normal --init-parms -1.0 1.0