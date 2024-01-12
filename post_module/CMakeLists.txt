cmake_minimum_required(VERSION 3.4...3.18)
project(post_module)

add_subdirectory(pybind11)
pybind11_add_module(post_module
    src/post.cpp
    src/helpers.h
)

find_package(OpenMP REQUIRED)
target_link_libraries(post_module PRIVATE OpenMP::OpenMP_CXX)