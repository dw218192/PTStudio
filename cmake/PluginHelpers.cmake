# ============================================================================
# Plugin Build Helpers
# ============================================================================
# This module provides CMake functions for building PTStudio plugins with
# consistent configuration and symbol visibility settings.

#
# pts_add_plugin
#
# Creates a plugin target with proper configuration for the PTStudio plugin system.
#
# Usage:
#   pts_add_plugin(
#       NAME <plugin_name>
#       SOURCES <source1> [<source2> ...]
#       [DEPENDENCIES <dep1> [<dep2> ...]]
#   )
#
# Parameters:
#   NAME         - Name of the plugin target (required)
#   SOURCES      - List of source files for the plugin (required)
#   DEPENDENCIES - Optional list of target dependencies (e.g., core::api, other libraries)
#
# The function automatically:
#   - Creates a shared library target
#   - Hides all symbols by default (only PTS_PLUGIN_EXPORT symbols are visible)
#   - Configures output directories to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
#   - Links core::api automatically
#
function(pts_add_plugin)
    # Parse arguments
    cmake_parse_arguments(
        PLUGIN                              # prefix
        ""                                  # options (flags)
        "NAME"                              # single-value arguments
        "SOURCES;DEPENDENCIES"              # multi-value arguments
        ${ARGN}                             # arguments to parse
    )

    # Validate required arguments
    if(NOT PLUGIN_NAME)
        message(FATAL_ERROR "pts_add_plugin: NAME is required")
    endif()
    if(NOT PLUGIN_SOURCES)
        message(FATAL_ERROR "pts_add_plugin: SOURCES is required")
    endif()

    # Create the shared library target
    add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SOURCES})

    # Link dependencies (always include core::api)
    target_link_libraries(${PLUGIN_NAME} PRIVATE core::api)
    if(PLUGIN_DEPENDENCIES)
        target_link_libraries(${PLUGIN_NAME} PRIVATE ${PLUGIN_DEPENDENCIES})
    endif()

    # Configure plugin target properties
    # Hide all symbols by default, only export those marked with PTS_PLUGIN_EXPORT
    # This prevents symbol conflicts between plugins and reduces the dynamic symbol table size
    set_target_properties(${PLUGIN_NAME} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
    )

    # Set output directories for all configurations
    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER ${config} config_upper)
        set_target_properties(${PLUGIN_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
            LIBRARY_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
        )
    endforeach()

    message(STATUS "Plugin configured: ${PLUGIN_NAME}")
endfunction()

