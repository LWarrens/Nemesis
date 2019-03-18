#pragma once

#include <unordered_map>
#include <vector>
#include "../Version.hpp"

namespace Nemesis {
	template<typename InputType = float, typename OutputType = float>
	struct TrainingInstance_t {
		TrainingInstance_t() {}
		TrainingInstance_t(std::vector<InputType> input, std::vector<OutputType> target) : input(input), target(target) {}
		std::vector<InputType> input;
		std::vector<OutputType> target;
	};
	typedef TrainingInstance_t<float, float> TrainingInstance;

	template<std::size_t n_inputs, std::size_t n_outputs, typename InputType, typename OutputType>
	struct Estimator {
		static const size_t num_inputs = n_inputs;
		static const size_t num_outputs = n_outputs;
		typedef InputType input_type;
		typedef OutputType output_type;
		typedef double ErrorType;
		virtual std::vector<OutputType> predict(std::vector<InputType> input) = 0;
		virtual ErrorType fit(std::vector<TrainingInstance_t<InputType, OutputType>> samples) = 0;
	};
}