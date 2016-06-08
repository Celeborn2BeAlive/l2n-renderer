set(GLFW_BUILD_DOCS off)
set(GLFW_BUILD_EXAMPLES off)
set(GLFW_BUILD_TESTS off)
set(GLFW_BUILD_INSTALL off)
add_subdirectory(glfw)

set(GLM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glm)
add_subdirectory(c2ba/maths)

add_subdirectory(c2ba/glutils)