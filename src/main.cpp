#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glad/glad.h>

#define C2BA_GLUTILS_GL3_HEADER <glad/glad.h>
#include <c2ba/glutils/GLBuffer.hpp>
#include <c2ba/glutils/GLVertexArray.hpp>
#include <c2ba/glutils/GLShader.hpp>
#include <c2ba/glutils/GLProgram.hpp>

namespace l2n
{

const char* vertexShaderSource =
    "#version 330\n"
    "layout(location = 0) in vec3 iVertexPosition;"
    "layout(location = 1) in vec3 iVertexColor;"
    "out vec3 FragColor;"
    "void main() {"
    "FragColor = iVertexColor;"
    "gl_Position = vec4(iVertexPosition, 1.f);"
"}";

const char* fragmentShaderSource =
    "#version 330\n"
    "in vec3 FragColor;"
    "out vec4 oFragColor;"
    "void main() {"
    "oFragColor = vec4(FragColor, 1.f);"
"}";

int main(int argc, char** argv)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    { 
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    if (!gladLoadGL()) {
        printf("Something went wrong!\n");
        exit(-1);
    }

    GLfloat vertices[] = {
        -0.5, -0.5,/* Position */ 1., 0., 0., /* Couleur */ // Premier vertex
        0.5, -0.5,/* Position */ 0., 1., 0., /* Couleur */ // Deuxième vertex
        0., 0.5,/* Position */ 0., 0., 1. /* Couleur */ // Troisème vertex
    };

    c2ba::GLBufferStorage<GLfloat> vbo{ sizeof(vertices) / sizeof(float), vertices };
    c2ba::GLVertexArray vao;

    vao.enableVertexAttrib(0);
    vao.vertexAttribOffset(vbo.glId(), 0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), 0);

    vao.enableVertexAttrib(1);
    vao.vertexAttribOffset(vbo.glId(), 1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), 2 * sizeof(GLfloat));

    c2ba::GLProgram program;

    {
        c2ba::GLShader vertexShader{ GL_VERTEX_SHADER };
        vertexShader.setSource(vertexShaderSource);
        vertexShader.compile();
        if (!vertexShader.getCompileStatus()) {
            std::cerr << "Vertex Shader error:" << vertexShader.getInfoLog() << std::endl;
        }

        c2ba::GLShader fragmentShader{ GL_FRAGMENT_SHADER };
        fragmentShader.setSource(fragmentShaderSource);
        fragmentShader.compile();
        if (!fragmentShader.getCompileStatus()) {
            std::cerr << "Fragment Shader error:" << fragmentShader.getInfoLog() << std::endl;
        }

        program.attachShader(vertexShader);
        program.attachShader(fragmentShader);
        program.link();
        if (!program.getLinkStatus()) {
            std::cerr << "Program link error:" << program.getInfoLog() << std::endl;
        }
    }

    program.use();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        vao.bind();

        glDrawArrays(GL_TRIANGLES, 0 /* Pas d'offset au début du VBO */, 3);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

}

int main(int argc, char** argv)
{
    return l2n::main(argc, argv);
}