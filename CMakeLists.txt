cmake_minimum_required(VERSION 2.8)

project(l2n-renderer)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(c2ba-glsl-shaders)
include(c2ba-copy-dll-post-build)

file(
    GLOB_RECURSE
    L2N_SRC_FILES
    *.cpp *.hpp *.glsl
)

add_executable(
    l2n
    ${L2N_SRC_FILES}
)