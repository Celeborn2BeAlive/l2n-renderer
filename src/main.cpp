#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glad/glad.h>

#define C2BA_GLUTILS_GL3_HEADER <glad/glad.h>
#include <c2ba/glutils/GLBuffer.hpp>
#include <c2ba/glutils/GLVertexArray.hpp>
#include <c2ba/glutils/GLShader.hpp>
#include <c2ba/glutils/GLProgram.hpp>

#include <filesystem>

namespace l2n
{

namespace fs = std::experimental::filesystem;

int main(int argc, char** argv)
{
    fs::path appPath{ argv[0] };
    auto appDir = appPath.parent_path();

    if (!glfwInit()) {
        std::cerr << "Unable to init GLFW.\n";
        return -1;
    }
        
    auto window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window) { 
        std::cerr << "Unable to open window.\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGL()) {
        std::cerr << "Unable to init OpenGL.\n";
        return -1;
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
        vertexShader.setSource(c2ba::loadShaderSource(appDir / "glsl/triangle.vs.glsl"));
        vertexShader.compile();
        if (!vertexShader.getCompileStatus()) {
            std::cerr << "Vertex Shader error:" << vertexShader.getInfoLog() << std::endl;
        }

        c2ba::GLShader fragmentShader{ GL_FRAGMENT_SHADER };
        fragmentShader.setSource(c2ba::loadShaderSource(appDir / "glsl/triangle.fs.glsl"));
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