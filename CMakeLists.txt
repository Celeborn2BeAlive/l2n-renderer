cmake_minimum_required(VERSION 2.8)

project(l2n-renderer)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(third-party/submodules)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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

target_link_libraries(
    l2n
    glfw
)

target_include_directories(
    l2n
    PUBLIC third-party/submodules/glfw/include
)

install(
    TARGETS l2n
    DESTINATION bin
)