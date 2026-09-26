cmake_minimum_required(VERSION 3.20)
project(dlss5_scene_capture CXX)

# ReShade's add-on headers (BSD-3, crosire/reshade include/). They are not
# vendored: scripts/build_capture.ps1 fetches the tag that matches the ReShade
# players run, because an add-on built against newer headers than the running
# ReShade is refused at load.
set(RESHADE_INCLUDE "" CACHE PATH "Folder containing reshade.hpp")
if(NOT EXISTS "${RESHADE_INCLUDE}/reshade.hpp")
    message(FATAL_ERROR "Set RESHADE_INCLUDE to the folder containing reshade.hpp")
endif()
# ImGui header matching the ReShade release exactly (reshade_overlay.hpp
# refuses any other version), for the on-screen "capturing / complete" banner.
set(IMGUI_INCLUDE "" CACHE PATH "Folder containing imgui.h")
if(NOT EXISTS "${IMGUI_INCLUDE}/imgui.h")
    message(FATAL_ERROR "Set IMGUI_INCLUDE to the folder containing imgui.h")
endif()

add_library(dlss5_scene_capture SHARED scene_capture.cpp)
target_include_directories(dlss5_scene_capture PRIVATE "${RESHADE_INCLUDE}" "${IMGUI_INCLUDE}")
target_compile_features(dlss5_scene_capture PRIVATE cxx_std_17)
# windowscodecs: PNG encoding; ole32: COM; shell32: folder picker and "Open".
target_link_libraries(dlss5_scene_capture PRIVATE windowscodecs ole32 shell32)
target_compile_definitions(dlss5_scene_capture PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
# Static CRT: the add-on is dropped into a game folder, where the right VC++
# runtime may not be installed, and a missing DLL fails silently in ReShade.
set_property(TARGET dlss5_scene_capture PROPERTY
             MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ReShade only loads files named *.addon64 (64-bit games).
set_target_properties(dlss5_scene_capture PROPERTIES
    OUTPUT_NAME "dlss5_scene_capture"
    SUFFIX ".addon64"
    PREFIX "")
