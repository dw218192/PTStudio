# Manually add imgui
add_library(imgui STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_tables.cpp
)
# add windows, linux and mac specific files
# use glfw as backend
file(GLOB IMGUI_BACKEND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.*
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.*
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_win32.*
)
target_sources(imgui 
    PRIVATE
        ${IMGUI_BACKEND_SOURCES}
)
target_include_directories(imgui
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends
)
target_link_libraries(imgui
    PRIVATE
        glfw
)

# Manually add imguizmo
add_library(imguizmo STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/GraphEditor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/ImCurveEdit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/ImGradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/ImGuizmo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/ImSequencer.cpp
)
target_include_directories(imguizmo
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imguizmo
)
target_link_libraries(imguizmo
    PUBLIC
        imgui
)

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
        imgui
)