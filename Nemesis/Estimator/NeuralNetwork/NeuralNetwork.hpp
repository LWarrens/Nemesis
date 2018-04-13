#pragma once

#include <functional>
#include "../../Math.hpp"
#include "../Estimator.hpp"
#include "Activation.hpp"
namespace Nemesis {
	struct Cell {};

	template<typename WeightType>
	struct Neuron : Cell {
		std::vector<WeightType> weights;
		WeightType bias;
		Activation<WeightType> activation;
		//weight update rule?
		WeightType activate(std::vector<WeightType> neuron_input) {
			return bias + dot(weights, neuron_input);
		}

		void update(WeightType learning_rate, WeightType delta, std::vector<WeightType> neuron_input) {
			WeightType change = learning_rate * delta;
			weights += neuron_input * change;
			bias += change;
		}
	};

	template<std::size_t n_inputs, std::size_t n_outputs, typename InputType, typename OutputType>
	struct NeuralNetwork : Estimator<n_inputs, n_outputs, InputType, OutputType> {
		virtual std::vector<OutputType> predict(std::vector<InputType> input) = 0;
		virtual ErrorType fit(std::vector<TrainingInstance<InputType, OutputType>> sample) = 0;
	};
}