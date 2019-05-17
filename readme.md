# disentangled attribution curves (DAC)
Official code for using / reproducing DAC from the paper "Disentangled Attribution Curves for Interpreting Random Forests"



# documentation

## using DAC on new models
- the core of the method code lies in the [DAC](dac) folder and is compatible with scikit-learn
- the [examples/xor_dac.ipynb](examples/simple_ex.py) folder contains examples of how to use DAC on a new dataset with some simple datasets (e.g. XOR, etc.)
- the basic api is the function ```dac()``` which takes in an rf and returns a DAC curve
-```dac``` inputs:
forest: an sklearn ensemble of decision trees
input_space_x: the matrix of training data (feature values), a numpy 2D array
outcome_space_y: the array of training data (labels/regression targets), a numpy 1D array
assignment: a matrix of feature values that will have their DAC importance score evaluated, a numpy 2D array
S: a binary indicator of whether to include each feature in the importance calculation, a numpy 1D array with values 0 and 1 only
continuous_y: a boolean indicator of whether the y targets are regression(true) or classification(false), defaults to true
class_id: if classification, the class value to return proportions for, defaults to 1
-```dac``` outputs:
for regression: a numpy array whose length corresponds to the number of samples in the assignment input.  Each entry is a DAC importance score, a
float between min(outcome_space_y) and max(outcome_space_y)
for classification: a numpy array whose length corresponds to the number of samples in the assignment input.  Each entry is a DAC importance score, a
float between 0 and 1
-```dac_plot``` inputs:
forest: an sklearn ensemble of decision trees (random forest or adaboosted forest)
input_space_x: the matrix of training data (feature values), a numpy 2D array
outcome_space_y: the array of training data (labels/regression targets), a numpy 1D array
S: a binary indicator of whether to include each feature in the importance calculation, a numpy 1D array with values 0 and 1 only
interval_x: an interval for the x axis of the plot, defaults to None.  If None, a reasonable interval will be extrapolated from the range
of the first feature specified in S.
interval_y: an interval for the y axis of the plot (only applicable to heat maps), defaults to None.  
If None, a reasonable interval will be extrapolated from the range of the second feature specified in S.
di_x: a step length for the x axis of the plot, defaults to None.  If None, a reasonable step length will be extrapolated from the range
of the first feature specified in S.
di_y: a step length for the y axis of the plot (only applicable to heat maps), defaults to None.  If None, a reasonable step
length will be extrapolated from the range of the second feature specified in S.
C: a hyper-parameter specifying the number of standard deviations samples can be from the mean of the leaf and be counted into the curve.  Smaller values
yield a more sensitive curve, larger values yield a smoother curve.
continuous_y: a boolean indicator of whether the y targets are regression(true) or classification(false), defaults to true
weights: weights for the individual estimators contributions to the curve, defaults to None.  If None, weights will be extrapolated from the forest type.
-```dac_plot``` outputs:
a numpy array containing values for the DAC curve or heatmap describing the interaction of the variables specified in S

## reproducing results from the paper
- the [examples/bike_sharing_dac.ipynb](examples/bike_sharing_dac.ipynb) folder contains examples of how to use DAC to reproducing the qualitative curves on the bike-sharing dataset in the paper
- the [simulation script](experiments/simulation/run_sim_synthetic.py) replicates the experiments with running simulations
- the [pmlb script](experiments/pmlb/run_dac_feature_engineered.py) replicates the experiments of automatic feature engineering on pmlb datasets


# todo before submitting
- remove scripts, eda folders, analyze nbs
