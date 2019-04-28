import itertools
from slurmpy import Slurm

partition = 'low'

# sweep different ways to initialize weights
dset_nums = list(range(163)) # there are 163 total

# run
s = Slurm("pmlb", {"partition": partition, "time": "3-0", "mem": "MaxMemPerNode"})

# iterate
for i in range(len(dset_nums)):
    param_str = 'module load python; python3 /accounts/projects/vision/chandan/rf_interactions/pmlb_comparisons/alt_run.py '
    param_str += str(dset_nums[i])
    s.run(param_str)
