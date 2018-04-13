#pragma once
#include <random>
#include <ctime>
#include "Version.hpp"
std::mt19937 rng((unsigned int)std::time(0));
