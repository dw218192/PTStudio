cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(plugin_name "editorRenderer")

pts_add_plugin(
    NAME ${plugin_name}
    SOURCES editorRenderer.cpp plugin.cpp
)