#pragma once
#include <ctgmath>

namespace Nemesis {
	template<typename WeightType = float >
	struct Activation {
		std::string type;
		virtual WeightType function(WeightType&) = 0;
		virtual WeightType derivative_function(WeightType&) = 0;
	};

	template<typename WeightType = float>
	struct LinearActivation_t : Activation<WeightType> {
		LinearActivation_t() {
			type = "linear";
		}
		WeightType function(WeightType& x) {
			return x;
		}		
		WeightType derivative_function(WeightType& x) {
			return 1;
		}
	};

	typedef LinearActivation_t<float> LinearActivation;

	template<typename WeightType = float>
	struct TanhActivation_t : Activation<WeightType> {
		TanhActivation_t() {
			type = "tanh";
		}
		WeightType function(WeightType& x) {
			return tanh(x);
		}
		WeightType derivative_function(WeightType& x) {
			return 1.0 - pow(tanh(x), 2);
		}
	};

	typedef TanhActivation_t<float> TanhActivation;

	template<typename WeightType = float>
	struct LogisticActivation_t : Activation<WeightType> {
		LogisticActivation_t() {
			type = "logistic";
		}
		WeightType function(WeightType& x) {
			return 1.0 / (1.0 + exp(-x));
		}
		WeightType derivative_function(WeightType& x) {
			auto e_x = exp(-x);
			return e_x / pow(1 + e_x, 2);
		}
	};

	typedef LogisticActivation_t<float> LogisticActivation;

	template<typename WeightType = float>
	struct SoftplusActivation_t : Activation<WeightType> {
		SoftplusActivation_t() {
			type = "softplus";
		}
		WeightType function(WeightType& x) {
			return log(1.0 + exp(-x));
		}
		WeightType derivative_function(WeightType& x) {
			return 1.0 / (1.0 + exp(-x));
		}
	};

	typedef SoftplusActivation_t<float> SoftplusActivation;

	template<typename WeightType = float>
	struct RectifierActivation_t : Activation<WeightType> {
		WeightType leaky_parameter = 1;
		RectifierActivation_t() {
			type = "rectifier";
		}
		RectifierActivation_t(WeightType parameter) : leaky_parameter(parameter) {
			type = "rectifier";
		}
		WeightType function(WeightType& x) {
			return max(leaky_parameter * x, x);
		}
		WeightType derivative_function(WeightType& x) {
			return leaky_parameter;
		}
	};

	typedef RectifierActivation_t<float> RectifierActivation;
}