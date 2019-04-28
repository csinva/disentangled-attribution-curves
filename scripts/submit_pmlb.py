import itertools
from slurmpy import Slurm
import pmlb as dsets



partition = 'low'

# sweep different ways to initialize weights
task = 'classification'
dset_names = dsets.classification_dataset_names

# task = 'regression'
# dset_names = dsets.regression_dataset_names

# run
s = Slurm("pmlb", {"partition": partition, "time": "3-0", "mem": "MaxMemPerNode"})

# iterate
for i in range(len(dset_names)):
    param_str = 'module load python; python3 /accounts/projects/vision/chandan/rf_interactions/pmlb_comparisons/alt_run.py '
    param_str += str(dset_names[i])
    param_str += ' ' + task
    s.run(param_str)
