cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(plugin_name "TestPlugin")

add_library(${plugin_name} SHARED
    testPlugin.cpp
)
target_link_libraries(${plugin_name}
PRIVATE
    core::api
)
set_target_properties(${plugin_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
)
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${config} config_upper)
    set_target_properties(${plugin_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
    )
endforeach()

