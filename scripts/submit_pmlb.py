import itertools
from slurmpy import Slurm
import pmlb as dsets

partition = 'high'
func_nums = range(1, 11)
# seeds = range(1)

# run
s = Slurm("pmlb", {"partition": partition, "time": "3-0", "mem": "MaxMemPerNode"})

# iterate
for i in range(len(func_nums)):
    param_str = 'module load python; python3 /accounts/projects/vision/chandan/rf_interactions/sim_comparisons/run_sim.py '
    param_str += str(func_nums[i])
    s.run(param_str)
