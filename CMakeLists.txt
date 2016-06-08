cmake_minimum_required(VERSION 2.8)

project(l2n-renderer)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(C2BA_GLUTILS_GL3_HEADER "<glad/glad.h>")
set(C2BA_GLUTILS_GL3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third-party/glad/include)
add_definitions(-DC2BA_GLUTILS_GL3_HEADER=${C2BA_GLUTILS_GL3_HEADER})
add_subdirectory(third-party/submodules)

find_package(OpenGL REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(c2ba-glsl-shaders)
include(c2ba-copy-dll-post-build)

file(
    GLOB_RECURSE
    L2N_SRC_FILES
    src/*.cpp src/*.hpp src/*.glsl
    third-party/glad/src/*.c
    third-party/submodules/imgui/imgui.cpp third-party/submodules/imgui/imgui_demo.cpp third-party/submodules/imgui/imgui_draw.cpp
)

c2ba_add_shader_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/shaders glsl)

add_executable(
    l2n
    ${L2N_SRC_FILES}
)

target_link_libraries(
    l2n
    ${OPENGL_LIBRARIES}
    glfw
    c2ba_maths
    c2ba_glutils
)

target_include_directories(
    l2n
    PUBLIC 
    ${OPENGL_INCLUDE_DIRS}
    third-party/submodules/glfw/include
    third-party/submodules/glm
    third-party/glad/include
    third-party/submodules/c2ba/maths/lib/include
    third-party/submodules/c2ba/glutils/lib/include
    third-party/submodules/
)

source_group ("glsl" REGULAR_EXPRESSION "*.glsl")
source_group ("imgui" REGULAR_EXPRESSION "third-party/submodules/imgui/*.*")

install(
    TARGETS l2n
    DESTINATION bin
)