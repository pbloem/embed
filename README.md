# embed

Implementation of basic KG embedding methods. Meant both as an example implementation and as a testbed for simple improvements.

## Running

The following command runs a simple experiment on the toy data (run from the reporisoty root):
```bash
python experiments/lp.bias.py -D fb -E 256 -e 10 --eval-int 1 -N 1 0 1 -l 0.1 -B 1024 --test-batch 100 -m distmult --opt adagrad --loss ce
```