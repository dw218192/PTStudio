# PTStudio Emscripten toolchain wrapper

if (NOT DEFINED ENV{EMSCRIPTEN})
    message(FATAL_ERROR "EMSCRIPTEN environment variable is not set.")
endif()

set(EMSCRIPTEN_ROOT "$ENV{EMSCRIPTEN}")
include("${EMSCRIPTEN_ROOT}/cmake/Modules/Platform/Emscripten.cmake")

set(PTS_STATIC_PLUGINS ON CACHE BOOL "" FORCE)
set(PTS_WINDOWING "glfw" CACHE STRING "" FORCE)
