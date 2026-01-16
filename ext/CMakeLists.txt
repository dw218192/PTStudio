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
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_vulkan.*
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
        Vulkan::Vulkan
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

# Manually add nativefiledialog
include(CheckIncludeFile)
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/nativefiledialog/src/nfd_common.c)

# add specific implementations
if (WIN32)
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/nativefiledialog/src/nfd_win.cpp)
elseif (APPLE)
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/nativefiledialog/src/nfd_cocoa.m)
elseif (UNIX)
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/nativefiledialog/src/nfd_gtk.c)
else ()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif ()

add_library(nativefiledialog ${SOURCES})

target_include_directories(nativefiledialog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/nativefiledialog/src/include)

# stb project (uses headers from Conan's stb package)
add_subdirectory(stb)
