#pragma once

#include <memory>
#include <functional>
#include "../Estimator/Estimator.hpp"
#include "../Version.hpp"

template<typename EstimatorType,
	typename = typename std::enable_if<std::is_base_of<Estimator<EstimatorType::num_outputs, EstimatorType::num_inputs, typename EstimatorType::input_type, typename EstimatorType::output_type>, EstimatorType>::value>::type>
	struct TDLearner {
	/*
	 * represents the Utility/Value function V[state, action(s)] = reward(s)
	 */
	std::unique_ptr<EstimatorType> q_estimator;
	std::function<void()> choose_action;

	TDLearner() : q_estimator(new EstimatorType()) {}

	TDLearner(EstimatorType estimator) : q_estimator(new EstimatorType(estimator)) {}

	void set_value_estimator(EstimatorType estimator) {
		q_estimator.reset(new EstimatorType(estimator));
	}

	/**
	 *
	 * @param state_action state data
	 * @param reward reward of actions performed
	 * @param action_indices indices of rewards to be updated for performing a certain action
	 */
	void update_values(std::vector<typename EstimatorType::input_type> state_action,
					   std::vector<typename EstimatorType::output_type> reward,
					   std::vector<size_t> action_indices) {
		std::vector<typename EstimatorType::output_type> old_reward = q_estimator->predict(state_action);
		for (auto index : action_indices) {
			old_reward[index] = reward[index];
		}
		std::vector<TrainingInstance_t<EstimatorType::input_type, EstimatorType::output_type>> samples {
			TrainingInstance<EstimatorType::input_type, EstimatorType::output_type>(state_action, old_reward)
		};
		q_estimator->fit(samples);
	}

	std::vector<typename EstimatorType::input_type>
		get_value(std::vector<typename EstimatorType::input_type> state_action) {
		return q_estimator->predict(state_action);
	}
};
