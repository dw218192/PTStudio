# Manually add ImGuiColorTextEdit
add_library(ImGuiColorTextEdit STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiColorTextEdit/TextEditor.cpp
)
target_include_directories(ImGuiColorTextEdit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiColorTextEdit
)
target_link_libraries(ImGuiColorTextEdit
    PUBLIC
        imgui::imgui
)

# stb project (uses headers from Conan's stb package)
add_subdirectory(stb)
