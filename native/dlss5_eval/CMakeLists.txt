cmake_minimum_required(VERSION 3.24)
project(dlss5_eval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The NVIDIA DLSS SDK, cloned by scripts/build_native.ps1. Not vendored: it is
# large, it moves, and it is NVIDIA's to distribute.
set(DLSS_SDK "${CMAKE_CURRENT_SOURCE_DIR}/../_sdk/DLSS" CACHE PATH "NVIDIA DLSS SDK root")

if(NOT EXISTS "${DLSS_SDK}/include/nvsdk_ngx.h")
    message(FATAL_ERROR
        "DLSS SDK headers not found under ${DLSS_SDK}. "
        "Run scripts\\build_native.ps1 rather than invoking CMake directly.")
endif()

add_executable(dlss5_eval main.cpp)

target_include_directories(dlss5_eval PRIVATE "${DLSS_SDK}/include")

# The static NGX helper. It dynamically loads nvngx_dlss*.dll at run time from
# the executable's own directory, which is why runtime.stage_runtime copies the
# user's DLLs next to the built binary instead of leaving them where they were.
target_link_directories(dlss5_eval PRIVATE "${DLSS_SDK}/lib/Windows_x86_64/x64")
target_link_libraries(dlss5_eval PRIVATE
    nvsdk_ngx_s
    d3d12
    dxgi
    dxguid
    user32
)

# /MT so the harness does not depend on a VC++ redistributable being installed.
# It gets copied around next to loose DLLs and must run from anywhere.
set_property(TARGET dlss5_eval PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")

set_target_properties(dlss5_eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
)
