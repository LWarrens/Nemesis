#pragma once
#include <vector>
#include "../../Math.hpp"
namespace Nemesis {


  template <typename WeightType>
  struct CostFunction {
		virtual std::vector<WeightType> get_cost(std::vector<WeightType>& target, std::vector<WeightType>& value) = 0;
	  virtual std::vector<WeightType> get_gradient(std::vector<WeightType>& target, std::vector<WeightType>& value) = 0;
  };

  template <typename WeightType>
  struct QuadraticCostFunction : CostFunction<WeightType> {
	  std::vector<WeightType> get_cost(std::vector<WeightType>& target, std::vector<WeightType>& value) {
		  auto diff = subtract(target, value);
		  return multiply(multiply(diff, diff), (WeightType).5);
	  }
	  std::vector<WeightType> get_gradient(std::vector<WeightType>& target, std::vector<WeightType>& value) {
		  return subtract(target, value);
	  }
  };
	//todo:fix
	template <typename WeightType>
	struct QuadraticCostFunctionLasso : CostFunction<WeightType> {
		WeightType lambda = .1;
		std::vector<WeightType> get_cost(std::vector<WeightType>& target, std::vector<WeightType>& value) {
			auto diff = subtract(target, value);
			auto sum = std::accumulate(std::begin(diff), std::end(diff), (WeightType)0);
			return multiply(multiply(diff, diff), (WeightType).5) + lambda * std::abs(sum) / diff.size();
		}
		std::vector<WeightType> get_gradient(std::vector<WeightType>& target, std::vector<WeightType>& value) {
			auto diff = subtract(target, value);
			auto sum = std::accumulate(std::begin(diff), std::end(diff), (WeightType)0);
			return diff + lambda * ((sum > 0) - (sum < 0)) / diff.size();
		}
	};
	template <typename WeightType>
	struct QuadraticCostFunctionRidge: CostFunction<WeightType> {
		WeightType lambda = .1;
		std::vector<WeightType> get_cost(std::vector<WeightType>& target, std::vector<WeightType>& value) {
			auto diff = subtract(target, value);
			auto diff2 = multiply(diff, diff);
			return multiply(diff2, (WeightType).5) + std::accumulate(std::begin(diff2), std::end(diff2), (WeightType)0) * lambda / (2 * diff2.size());
		}
		std::vector<WeightType> get_gradient(std::vector<WeightType>& target, std::vector<WeightType>& value) {
			auto diff = subtract(target, value);
			return diff + std::accumulate(std::begin(diff), std::end(diff), (WeightType)0) * lambda / diff.size();
		}
	};
}
