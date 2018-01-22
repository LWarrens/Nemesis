#pragma once

#include <functional>
#include "../../Math.hpp"
#include "../Estimator.hpp"
#include "Activation.hpp"

template<typename WeightType>
struct Neuron {
	std::vector<WeightType> weights;
	WeightType bias;
	Activation<WeightType> activation;
	//weight update rule?
	std::array<WeightType, 2> activate(std::vector<WeightType> neuron_input) {
		WeightType activation_value = bias + dot(weights, neuron_input);
		return {
				activation.function(activation_value),
				activation.derivative_function(activation_value)
		};
	}

	void update(WeightType learning_rate, WeightType delta, std::vector<WeightType> neuron_input) {
		WeightType change = learning_rate * delta;
		weights = add(weights, multiply(change, neuron_input));
		bias += change;
	}
};

template<typename InputType, typename OutputType>
struct NeuralNetwork : Estimator<InputType, OutputType> {
	virtual std::vector<OutputType> predict(std::vector<InputType> input) = 0;
	virtual double fit(std::vector<TrainingInstance<InputType, OutputType>> sample) = 0;
};
