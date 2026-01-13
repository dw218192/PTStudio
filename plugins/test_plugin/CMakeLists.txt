cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(plugin_name "test_plugin")

# Create shared library (DLL)
add_library(${plugin_name} SHARED
    test_plugin.cpp
)

# Link against core headers (for plugin API)
target_link_libraries(${plugin_name}
PRIVATE
    core
)

# Set output directory to bin/ (same as main executable)
set_target_properties(${plugin_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# Ensure consistent output for multi-config generators
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${config} config_upper)
    set_target_properties(${plugin_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
endforeach()

