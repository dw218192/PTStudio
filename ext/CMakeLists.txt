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
