# Nemesis
Feed Forward Neural Networks for Reinforcement Learning (Soon to be extended)


## Estimator
Estimator is a general interface for estimators.
An estimator consist of:
  * an input type and output type that define the expected type of input vector values and output vector values.
  * an expected number of input and output attributes
  * **predict**; an estimator should produce an output from a given input vector using predict
  * **fit**; an estimator should be trained using fit

### MLP
The MLP class is a Multi-Layer Perceptron. The MLP class is a subclass of Estimator.

A simple example of how to construct this class is this,

MLP<float, float, float> (1, 1)

where each float type defines the input type, output type, and weight type respectively for the network.

The (1, 1) constructor constructs a neural network expecting 1 input and 1 output.

**learning_rate**: a variable that can be accessed directly; it represents the learning rate of the MLP

**decay_rate**: a variable that can be accessed directly; it represents the learning rate of the MLP

Currently, an MLP can be saved and loaded to a json file using the MLP's load and save functions.

## TDLearner
TDLearner is an interface for reinforcement learning agents using TD-learning methods.

It requires an Estimator to estimate and update the value of state action pairs.

**set_value_estimator**: replaces the current estimator with a new estimator

**update_values** function: updates the TD function for the state reward pair(s) (TODO: make update_values action_index a set)

**get_value**: gets the values for a state reward pair

## Todo:
* remove nhlohmann's json file and add as a requirement
