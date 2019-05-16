import itertools
from slurmpy import Slurm
import pmlb as dsets

partition = 'low'
dset_nums = range(120) # 120 total
# seeds = range(1)

# run
s = Slurm("real_sim", {"partition": partition, "time": "1-0", "mem": "MaxMemPerNode"})

# iterate
for i in range(len(dset_nums)):
    param_str = 'module load python; python3 /accounts/projects/vision/chandan/rf_interactions/sim_comparisons/run_sim_real_data.py '
    param_str += str(dset_nums[i])
    s.run(param_str)
