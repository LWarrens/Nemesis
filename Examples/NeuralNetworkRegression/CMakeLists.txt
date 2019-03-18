# 3.8 used for cxx_std_17 enumeration
cmake_minimum_required(VERSION 3.8)
# Add source to this project's executable.
include_directories(
"../../"
"../../External/nlohmann"
)
add_executable(NNTest "main.cpp")
# Use C++17
target_compile_features(NNTest PUBLIC cxx_std_17)
