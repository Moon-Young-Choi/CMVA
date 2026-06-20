cmake_minimum_required(VERSION 3.15)
project(cmva_cpp LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cmva_cpp src/cmva_cpp.cpp)
target_compile_features(cmva_cpp PRIVATE cxx_std_17)

install(TARGETS cmva_cpp LIBRARY DESTINATION .)
