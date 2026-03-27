cmake_minimum_required(VERSION 3.19)
project(usdz_pack CXX)

find_package(openusd REQUIRED)
add_executable(usdz_pack usdz_pack.cpp)
target_link_libraries(usdz_pack
    openusd::usd_usdUtils
    openusd::usd_sdf
    openusd::usd_ar
)
install(TARGETS usdz_pack RUNTIME DESTINATION bin)
