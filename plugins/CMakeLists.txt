add_library(pts_plugins INTERFACE)
add_library(pts::plugins ALIAS pts_plugins)

add_subdirectory(test_plugin)

if(PTS_STATIC_PLUGINS)
    get_property(static_plugin_targets GLOBAL PROPERTY PTS_STATIC_PLUGIN_TARGETS)
    if(static_plugin_targets)
        target_link_libraries(pts_plugins INTERFACE ${static_plugin_targets})
    endif()
endif()