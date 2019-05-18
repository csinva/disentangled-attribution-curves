import itertools
from slurmpy import Slurm
import pmlb as dsets

partition = 'high'
func_nums = range(1, 11)
seeds = range(11)

# run
s = Slurm("sim", {"partition": partition, "time": "3-0", "mem": "MaxMemPerNode"})

# iterate
for i in range(len(func_nums)):
    for j in range(len(seeds)):
        param_str = 'module load python; python3 /accounts/projects/vision/chandan/disentangled_attribution_curves/experiments/simulation/run_sim_synthetic.py '
        param_str += str(func_nums[i])
        param_str += ' ' + str(seeds[j])
        s.run(param_str)
