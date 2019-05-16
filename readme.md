# disentangled attribution curves (DAC)
Official code for using / reproducing DAC from the paper "Disentangled Attribution Curves for Interpreting Random Forests" 



# documentation

## using DAC on new models
- the core of the method code lies in the [DAC](dac) folder and is compatible with scikit-learn
- the [examples/xor_dac.ipynb](examples/simple_ex.py) folder contains examples of how to use DAC on a new dataset with some simple datasets (e.g. XOR, etc.)
- the basic api is the function ```dac()``` which takes in an rf and returns a DAC curve


## reproducing results from the paper
- the [examples/bike_sharing_dac.ipynb](examples/bike_sharing_dac.ipynb) folder contains examples of how to use DAC to reproducing the qualitative curves on the bike-sharing dataset in the paper
- the [simulation script](experiments/simulation/run_sim_synthetic.py) replicates the experiments with running simulations
- the [pmlb script](experiments/pmlb/run_dac_feature_engineered.py) replicates the experiments of automatic feature engineering on pmlb datasets


# todo before submitting
- remove scripts, eda folders