set(GLFW_BUILD_DOCS off)
set(GLFW_BUILD_EXAMPLES off)
set(GLFW_BUILD_TESTS off)
set(GLFW_BUILD_INSTALL off)
add_subdirectory(glfw)

set(GLM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glm)
add_subdirectory(c2ba/maths)

set(C2BA_GLUTILS_GL3_HEADER "<glad/glad.h>")
set(C2BA_GLUTILS_GL3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third-party/glad/include)
add_subdirectory(c2ba/glutils)