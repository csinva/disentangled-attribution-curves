# disentangled attribution curves (DAC)
Official code for using / reproducing DAC from the paper "Disentangled Attribution Curves for Interpreting Random Forests" 



# documentation

## using DAC on new models
- the core of the method code lies in the [DAC](dac) folder and is compatible with scikit-learn
- the [examples/simple_ex.ipynb](examples/simple_ex.py) folder contains examples of how to use DAC on a new dataset with some simple datasets (e.g. XOR, etc.)


## reproducing results from the paper
- the [examples/bike_sharing_curves.ipynb](examples/bike_sharing_curves.ipynb) folder contains examples of how to use DAC to reproducing the qualitative curves on the bike-sharing dataset in the paper
- the [script](script) in the simulation_experiments folder replicates the experiments with running simulations
- the [pmlb script](pmlb script) in the pmlb_experiments folder replicates the experiments of automatic feature engineering



# todo before submitting
- remove scripts, eda folders