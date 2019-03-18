#pragma once

#include <vector>
#include <stdexcept>
#include "Version.hpp"

namespace Nemesis {
	template<typename T>
	T dot(std::vector<T> a, const std::vector<T>& b) {
		if (a.size() == b.size() && !a.empty()) {
			a[0] *= b[0];
			for (size_t i = 1; i < a.size(); ++i) {
				a[0] += a[i] * b[i];
			}
			return a[0];
		}
		if (a.size() != b.size()) {
			throw std::runtime_error("dot product multiplication: vectors are not the same size");
		}
		throw std::runtime_error("vectors must be non-empty");
	}

	template<typename T>
	std::vector<T> multiply(std::vector<T> a, const std::vector<T>& b) {
		if (a.size() == b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] *= b[i];
			}
			return a;
		}
		throw std::runtime_error("multiply: vectors be of equal length");
	}

	template<typename T>
	std::vector<T> multiply(const T& a, std::vector<T> b) {
		if (b.size()) {
			for (size_t i = 0; i < b.size(); ++i) {
				b[i] *= a;
			}
			return b;
		}
		throw std::runtime_error("multiply: vector non-empty");
	}

	template<typename T>
	std::vector<T> multiply(const std::vector<T>& a, const T& b) {
		return multiply(b, a);
	}

	template<typename T>
	std::vector<T> operator*(std::vector<T> a, const T& b) {
		if (a.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] *= b;
			}
			return a;
		}
		throw std::runtime_error("multiply: vector non-empty");
	}

	template<typename T>
	std::vector<T> operator*(const T& a, std::vector<T> b) {
		if (b.size()) {
			for (size_t i = 0; i < b.size(); ++i) {
				b[i] *= a;
			}
			return b;
		}
		throw std::runtime_error("multiply: vector non-empty");
	}

	template<typename T>
	std::vector<T> operator*(std::vector<T> a, const std::vector<T>& b) {
		if (a.size() == b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] *= b[i];
			}
			return a;
		}
		throw std::runtime_error("multiply: vectors be of equal length");
	}

	template<typename T>
	std::vector<T> subtract(std::vector<T> a, const std::vector<T>& b) {
		if (a.size() == b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] -= b[i];
			}
			return a;
		}
		throw std::runtime_error("Subtract: vectors should have equal length");
	}


	template<typename T>
	std::vector<T> subtract(std::vector<T> a, const T& b) {
		if (b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] -= b;
			}
			return a;
		}
		throw std::runtime_error("multiply: vector non-empty");
	}

	template<typename T>
	std::vector<T> add(std::vector<T> a, const std::vector<T>& b) {
		if (a.size() == b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] += b[i];
			}
			return a;
		}
		throw std::runtime_error("Subtract: vectors should have equal length");
	}


	template<typename T>
	std::vector<T> add(std::vector<T> a, const T& b) {
		if (b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] += b;
			}
			return a;
		}
		throw std::runtime_error("multiply: vector non-empty");
	}

	template<typename T>
	std::vector<T> operator+(std::vector<T> a, const T& b) {
		if (a.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] += b;
			}
			return a;
		}
		throw std::runtime_error("multiply: vector non-empty");
	}

	template<typename T>
	std::vector<T> operator+(std::vector<T> a, const std::vector<T>& b) {
		if (a.size() == b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] += b[i];
			}
			return a;
		}
		throw std::runtime_error("Subtract: vectors should have equal length");
	}

	template<typename T>
	std::vector<T> operator+(const T& b, const std::vector<T>& a) {
		return a + b;
	}

	template<typename T>
	std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b) {
		a = a + b;
		return a;
	}


	template<typename T>
	std::vector<T>& operator+=(std::vector<T>& a, const T& b) {
		a = b + a;
		return a;
	}

	template<typename T>
	std::vector<T> operator-(std::vector<T> a, const std::vector<T>& b) {
		if (a.size() == b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] -= b[i];
			}
			return a;
		}
		throw std::runtime_error("Subtract: vectors should have equal length");
	}

	template<typename T>
	std::vector<T> operator-(const T& a, std::vector<T> b) {
		if (b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				b[i] = a - b[i];
			}
			return b;
		}
		throw std::runtime_error("multiply: vector non-empty");
	}

	template<typename T>
	std::vector<T> operator-(std::vector<T> b, const T& a) {
		if (b.size()) {
			for (size_t i = 0; i < a.size(); ++i) {
				b[i] -= a;
			}
			return b;
		}
		throw std::runtime_error("multiply: vector non-empty");
	}

	struct weight {

	};

	struct vector {

	};
}