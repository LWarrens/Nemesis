#pragma once
#include <random>
#include <ctime>
std::mt19937 rng((unsigned int)std::time(0));