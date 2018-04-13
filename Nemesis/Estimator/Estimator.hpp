#pragma once

#include <unordered_map>
#include <vector>
#include "../Version.hpp"

namespace Nemesis {
	template<typename InputType = float, typename OutputType = float>
	struct TrainingInstance {
		TrainingInstance() {}
		TrainingInstance(std::vector<InputType> input, std::vector<OutputType> target) : input(input), target(target) {}
		std::vector<InputType> input;
		std::vector<OutputType> target;
	};

	template<std::size_t n_inputs, std::size_t n_outputs, typename InputType, typename OutputType>
	struct Estimator {
		static const size_t num_inputs = n_inputs;
		static const size_t num_outputs = n_outputs;
		typedef InputType input_type;
		typedef OutputType output_type;
		typedef double ErrorType;
		virtual std::vector<OutputType> predict(std::vector<InputType> input) = 0;
		virtual ErrorType fit(std::vector<TrainingInstance<InputType, OutputType>> samples) = 0;
	};
}