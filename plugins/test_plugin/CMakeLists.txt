cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(plugin_name "TestPlugin")

pts_add_plugin(
    NAME ${plugin_name}
    SOURCES testPlugin.cpp
    DEPENDENCIES test_plugin::api
)

# Static variant used for static-plugin tests.
add_library(${plugin_name}Static STATIC testPlugin.cpp)
target_link_libraries(${plugin_name}Static PRIVATE core::api test_plugin::api)
target_compile_definitions(${plugin_name}Static PRIVATE PTS_STATIC_PLUGINS)

# ============================================================================
# Interface Target for public API
# ============================================================================
add_library(test_plugin_api INTERFACE)
add_library(test_plugin::api ALIAS test_plugin_api)

target_include_directories(test_plugin_api
INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/api
)