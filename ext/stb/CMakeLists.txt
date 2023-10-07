cmake_minimum_required(VERSION 3.2)
project(stb)

file(GLOB_RECURSE src_files 
  ${CMAKE_CURRENT_SOURCE_DIR}/stb_*.*
)

add_library(stb ${src_files})
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
