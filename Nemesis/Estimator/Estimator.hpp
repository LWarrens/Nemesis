#pragma once

#include <unordered_map>
#include <vector>
template<typename InputType = float, typename OutputType = float>
struct TrainingInstance {
	TrainingInstance() {}
	TrainingInstance(std::vector<InputType> input, std::vector<OutputType> target) : input(input), target(target) {}
	std::vector<InputType> input;
	std::vector<OutputType> target;
};

template<typename InputType, typename OutputType>
struct Estimator {
	int num_inputs;
	int num_outputs;
	typedef InputType input_type;
	typedef OutputType output_type;

	virtual std::vector<OutputType> predict(std::vector<InputType> input) = 0;

	//virtual double fit(std::vector<InputType> input, std::vector<OutputType> target) = 0;

	virtual double fit(std::vector<TrainingInstance<InputType, OutputType>> samples) = 0;
};