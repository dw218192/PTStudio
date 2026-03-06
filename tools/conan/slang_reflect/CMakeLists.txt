cmake_minimum_required(VERSION 3.20)
project(slang_reflect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(slang_reflect src/main.cpp)

target_include_directories(slang_reflect PRIVATE ${SLANG_INCLUDE_DIR})

if(WIN32)
    target_link_libraries(slang_reflect PRIVATE ${SLANG_LIB_DIR}/slang.lib)
else()
    target_link_libraries(slang_reflect PRIVATE ${SLANG_LIB_DIR}/libslang.so)
endif()

install(TARGETS slang_reflect DESTINATION bin)
