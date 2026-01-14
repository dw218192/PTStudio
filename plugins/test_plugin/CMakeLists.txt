cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(plugin_name "TestPlugin")

# Create the plugin target using the standardized helper function
pts_add_plugin(
    NAME ${plugin_name}
    SOURCES testPlugin.cpp
    DEPENDENCIES test_plugin::api
)

# ============================================================================
# Interface Target for public API
# ============================================================================
add_library(test_plugin_api INTERFACE)
add_library(test_plugin::api ALIAS test_plugin_api)

target_include_directories(test_plugin_api
INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/api
)