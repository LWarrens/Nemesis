#pragma once

#include <vector>
#include <array>
#include <ctime>
#include <random>
#include <stdexcept>
#include <json.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <execution>
#include <atomic>
#include "../../Random.hpp"
#include "NeuralNetwork.hpp"
#include "Activation.hpp"

using json = nlohmann::json;

namespace Nemesis {
	std::uniform_real_distribution<> dis(-1, 1);

	template<typename WeightType = float>
	struct Layer : std::vector<Neuron<WeightType>> {
		Layer() {}

		Layer(size_t size, Activation<WeightType>& activation) {
			add(size, activation);
		}

		Layer(std::vector<Neuron<WeightType>> layer_data) {
			assign(layer_data.begin(), layer_data.end());
		}

		Layer& add(size_t size, Activation<WeightType>& activation) {
			for (int i = 0; i < size; ++i) {
				Neuron<WeightType> a;
				a.activation = activation;
				this->push_back(a);
			}
			return *this;
		}
	};

	template<typename WeightType = float>
	struct Conv2DLayer : Layer<WeightType> {

	};

	template<std::size_t n_inputs, std::size_t n_outputs, typename InputType = float, typename OutputType = float, typename WeightType = float>
	struct MultiLayerPerceptron : NeuralNetwork<n_inputs, n_outputs, InputType, OutputType> {
		typedef std::vector<InputType> InputVectorType;
		typedef std::vector<WeightType> WeightVectorType;
		typedef std::vector<OutputType> OutputVectorType;
		MultiLayerPerceptron() {
		}
	private:
		WeightType learning_rate = .3;
		WeightType learning_decay = 0.001;
		unsigned long int learning_epoch = 0;

		std::vector<Layer<WeightType>> layers;
	public:
		double get_learning_rate() {
			return learning_rate;
		}
		void set_learning_rate(WeightType rate) {
			learning_rate = rate;
		}
		double get_learning_decay() {
			return learning_decay;
		}
		void set_learning_decay(WeightType decay) {
			learning_decay = decay;
		}
		double get_learning_epoch() {
			return learning_epoch;
		}
		void set_learning_epoch(unsigned int epoch = 0) {
			learning_epoch = epoch;
		}
		void append_layer(Layer<WeightType> layer) {
			auto num_weights = (layers.size() > 0) ? layers[layers.size() - 1].size() : num_inputs;
			std::uniform_real_distribution<> dis(-1, 1);
			for (Neuron<WeightType> &neuron : layer) {
				auto &weights = neuron.weights;
				weights.resize(num_weights);
				for (WeightType &weight : weights) {
					weight = dis(rng);
				}
				neuron.bias = dis(rng);
			}
			layers.push_back(layer);
		}

		void pop_layer() {
			layers.pop_back();
		}

		bool is_valid() {
			return this->num_outputs != layers[layers.size() - 1].size();
		}

		//todo: create full backprop form
		std::array<std::vector<WeightVectorType>, 2> propagate(InputVectorType input_data, float dropout) {
			if (layers.size() < 1) {
				throw std::exception("There must be at least 1 layer to propagate.");
			}
			//todo: explicitly show assumption that at least one layer exists
			static const int input_layer = 0;
			// create zero'd arrays of the same size
			std::vector<WeightVectorType> output(layers.size()), output_derivatives(layers.size());

			for (int i = 0; i < layers.size(); ++i) {
				output[i].resize(layers[i].size());
				output_derivatives[i].resize(layers[i].size());
			}

			// handle first layer special case where input is vector
			// convert input vector to correct type
			WeightVectorType input_data_weight(input_data.begin(), input_data.end());
			for (int j = 0; j < layers[0].size(); ++j) {
				auto& neuron = layers[input_layer][j];
				auto activation = neuron.activate(input_data_weight);
				output[input_layer][j] = neuron.activation.function(activation);
				output_derivatives[input_layer][j] = neuron.activation.derivative_function(activation);
			}
			// handle other cases with dynamic programming, input is the last layer
			for (int i = 1; i < layers.size(); ++i) {
				for (int j = 0; j < layers[i].size(); ++j) {
					auto& neuron = layers[i][j];
					auto activation = neuron.activate(output[i - 1]);
					output[i][j] = neuron.activation.function(activation);
					output_derivatives[i][j] = neuron.activation.derivative_function(activation);
				}
			}

			return { output, output_derivatives };
		}

		OutputVectorType predict(InputVectorType input) {
			if (layers.size() < 1) {
				throw std::exception("There must be at least 1 layer to propagate.");
			}
			//todo: explicitly show assumption that at least one layer exists
			static const int input_layer = 0;
			// create zero'd arrays of the same size

			// handle first layer special case where input is vector
			// convert input vector to correct type
			WeightVectorType inputs(input);
			WeightVectorType outputs(layers[0].size());
			for (int j = 0; j < layers[0].size(); ++j) {
				auto& neuron = layers[input_layer][j];
				//if fully connected just pass all inputs, if bitmask, do otherwise
				auto activation = neuron.activate(inputs);
				outputs[j] = neuron.activation.function(activation);
			}
			// handle other cases with dynamic programming, input is the last layer
			for (int i = 1; i < layers.size(); ++i) {
				inputs.swap(outputs);
				outputs.resize(layers[i].size());
				for (int j = 0; j < layers[i].size(); ++j) {
					auto& neuron = layers[i][j];
					//if fully connected just pass all inputs, if bitmask, do otherwise
					auto activation = neuron.activate(inputs);
					outputs[j] = neuron.activation.function(activation);
				}
			}
			return outputs;
		}

		ErrorType fit(std::vector<TrainingInstance<InputType, OutputType>> samples) {
			// DATA PARALLELIZABLE BLOCK
			const size_t sample_size = samples.size();
			if (sample_size > 0) {
				const size_t output_layer = layers.size() - 1;
				const double inverse_size = 1. / sample_size;
				std::vector<std::vector<WeightVectorType>> outputs(sample_size);
				std::vector<std::vector<WeightVectorType>> output_derivatives(sample_size);
				std::vector<std::vector<WeightVectorType>> sample_errors(sample_size, std::vector<WeightVectorType>(layers.size()));
				std::vector<WeightVectorType> average_outputs(layers.size());
				std::vector<WeightVectorType> average_output_derivatives(layers.size());
				std::vector<WeightVectorType> average_sample_error(layers.size());
				for (int i = 0; i < layers.size(); ++i) {
					average_outputs[i] = WeightVectorType(layers[i].size(), 0);
					average_output_derivatives[i] = WeightVectorType(layers[i].size(), 0);
					average_sample_error[i] = WeightVectorType(layers[i].size(), 0);
					for (int j = 0; j < sample_size; ++j) {
						sample_errors[j][i].resize(layers[i].size());
					}
				}

				std::atomic_int parallel_index;
				parallel_index.store(0);
				std::for_each(std::execution::par_unseq, samples.begin(), samples.end(), [this, &parallel_index, &samples, &outputs, &output_derivatives, &sample_errors, &output_layer](auto &given_sample) {
					int i = parallel_index.fetch_add(1);
					auto &sample = samples[i];
					// propagate
					std::array<std::vector<WeightVectorType>, 2> out_outd = propagate(sample.input, 0);
					outputs[i] = out_outd[0];
					output_derivatives[i] = out_outd[1];

					// Do error step for last layer
					auto difference = subtract(sample.target, outputs[i][outputs[i].size() - 1]);
					sample_errors[i][output_layer] = WeightVectorType(difference);
					for (int j = output_layer - 1; j > -1; --j) {
						auto next_layer = j + 1;
						for (int k = 0; k < layers[j].size(); ++k) {
							std::vector<WeightType> next_layer_weights(layers[next_layer].size());
							for (int w = 0; w < layers[next_layer].size(); ++w) {
								next_layer_weights[w] = layers[next_layer][w].weights[k];
							}
							sample_errors[i][j][k] = dot(next_layer_weights, sample_errors[i][next_layer]);
						}
					}
				});
				// END PARALLELIZABLE BLOCK

				for (int k = 0; k < sample_size; ++k) {
					auto decayed_rate = learning_rate / (1.f + learning_decay * learning_epoch++);
					for (int i = 0; i < average_outputs.size(); ++i) {
						for (int j = 0; j < average_outputs[i].size(); ++j) {
								average_outputs[i][j] += inverse_size * outputs[k][i][j];
								average_output_derivatives[i][j] += inverse_size * output_derivatives[k][i][j];
								average_sample_error[i][j] += inverse_size * sample_errors[k][i][j] * decayed_rate;
							}
						}
				}


				// update
				for (int i = 0; i < layers[0].size(); ++i) {
					WeightType delta = average_sample_error[0][i] * average_output_derivatives[0][i];
					for (int j_batch = 0; j_batch < samples.size(); ++j_batch) {
						layers[0][i].update(1, delta, samples[j_batch].input);
					}
				}
				for (int i = 1; i < layers.size(); ++i) {
					for (int j = 0; j < layers[i].size(); ++j) {
						WeightType delta = average_sample_error[i][j] * average_output_derivatives[i][j];
						layers[i][j].update(1, delta, average_outputs[i - 1]);
					}
				}
				//printf("actual: %f , target: %f\n", outputs[outputs.size()-1][0],target[0]);
			}
			return 0;
			//return backpropagate(samples, learning_rate);
		}

		void print_weights() {
			for (auto layer : layers) {
				std::printf("[");
				for (auto neuron : layer) {
					std::printf("{");
					const int last = neuron.weights.size() - 1;
					for (int i = 0; i < neuron.weights.size(); ++i) {
						if (i < last) {
							std::printf("%f,", neuron.weights[i]);
						}
						else {
							std::printf("%f; ", neuron.weights[i]);
						}
					}
					std::printf("%f}", neuron.bias);
				}
				std::printf("]\n");
			}
		}

		void json_load(json j_file) {
			std::vector<Layer<WeightType>> tmp_layers;
			for (auto j_layer : j_file["layers"]) {
				std::vector<Neuron<WeightType>> layer;
				for (auto j_neuron : j_layer["neurons"]) {
					Neuron<WeightType> n;
					for (auto j_weight : j_neuron["weights"]) {
						n.weights.push_back(j_weight.get<WeightType>());
					}
					n.bias = j_neuron["bias"].get<WeightType>();

					auto j_activation = j_neuron["activation"];
					std::string activation_type = j_activation["type"].get<std::string>();
					if (activation_type == "linear") {
						n.activation = LinearActivation<WeightType>();
					}
					else if (activation_type == "tanh") {
						n.activation = TanhActivation<WeightType>();
					}
					else if (activation_type == "logistic") {
						n.activation = LogisticActivation<WeightType>();
					}
					else if (activation_type == "softplus") {
						n.activation = SoftplusActivation<WeightType>();
					}
					else if (activation_type == "rectifier") {
						RectifierActivation<WeightType> recti;
						recti.leaky_parameter = j_activation["leaky_parameter"].get<WeightType>();
						n.activation = recti;
					}
					layer.push_back(n);
				}
				tmp_layers.push_back(layer);
			}
		}

		json as_json() {
			json j_file;
			for (auto layer : layers) {
				json j_layer;
				for (auto neuron : layer) {
					json j_neuron;
					j_neuron["weights"] = neuron.weights;
					j_neuron["bias"] = neuron.bias;
					j_neuron["activation"]["type"] = neuron.activation.type;
					j_layer["neurons"].push_back(j_neuron);
				}
				j_file["layers"].push_back(j_layer);
			}
			return j_file;
		}

		void load(const char* filename) {
			std::ifstream ifile(filename);
			if (ifile.good()) {
				json j_file;
				ifile >> j_file;
				std::vector<Layer<WeightType>> tmp_layers;
				for (auto &j_layer : j_file["layers"]) {
					std::vector<Neuron<WeightType>> layer;
					for (auto &j_neuron : j_layer["neurons"]) {
						Neuron<WeightType> n;
						for (auto j_weight : j_neuron["weights"]) {
							n.weights.push_back(j_weight.get<WeightType>());
						}
						n.bias = j_neuron["bias"].get<WeightType>();

						auto &j_activation = j_neuron["activation"];
						std::string activation_type = j_activation["type"].get<std::string>();
						if (activation_type == "linear") {
							n.activation = LinearActivation<WeightType>();
						}
						else if (activation_type == "tanh") {
							n.activation = TanhActivation<WeightType>();
						}
						else if (activation_type == "logistic") {
							n.activation = LogisticActivation<WeightType>();
						}
						else if (activation_type == "softplus") {
							n.activation = SoftplusActivation<WeightType>();
						}
						else if (activation_type == "rectifier") {
							RectifierActivation<WeightType> recti;
							recti.leaky_parameter = j_activation["leaky_parameter"].get<WeightType>();
							n.activation = recti;
						}
						layer.push_back(n);
					}
					tmp_layers.push_back(layer);
				}
				layers = tmp_layers;
			}
			else {
				//make better error catching
				std::printf("Error: File \"%s\" does not exist.\n", filename);
			}
		}

		void save(const char* filename) {
			json j_file;
			for (auto layer : layers) {
				json j_layer;
				for (auto neuron : layer) {
					json j_neuron;
					j_neuron["weights"] = neuron.weights;
					j_neuron["bias"] = neuron.bias;
					j_neuron["activation"]["type"] = neuron.activation.type;
					j_layer["neurons"].push_back(j_neuron);
				}
				j_file["layers"].push_back(j_layer);
			}
			std::ofstream ofile(filename);
			ofile << j_file;
			ofile.close();
		}

	};
}