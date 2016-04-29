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
#include <unordered_map>

namespace fs = std::experimental::filesystem;

namespace std {
template <> struct hash<fs::path>
{
    size_t operator()(const fs::path& x) const
    {
        using StringType = fs::path::string_type;
        return hash<StringType>()(x.c_str());
    }
};
}

namespace l2n
{

// Use .at() to access elements since GLShader has no default constructor
using ShaderLibrary = std::unordered_map<fs::path, c2ba::GLShader>;

c2ba::GLShader loadShader(const fs::path& shaderPath)
{
    using StringType = fs::path;
    static auto extToShaderType = std::unordered_map<StringType, std::pair<GLenum, std::string>>({
        { ".vs", { GL_VERTEX_SHADER, "vertex" } },
        { ".fs", { GL_FRAGMENT_SHADER, "fragment" } }
    });

    auto ext = shaderPath.stem().extension();
    auto it = extToShaderType.find(ext);
    if (it == end(extToShaderType)) {
        std::cerr << "Unrecognized shader extension " << ext << std::endl;
        throw std::runtime_error("Unrecognized shader extension " + ext.string());
    }

    std::clog << "Compiling " << (*it).second.second << " shader " << shaderPath << "\n";

    c2ba::GLShader shader{ (*it).second.first };
    shader.setSource(c2ba::loadShaderSource(shaderPath));
    shader.compile();
    if (!shader.getCompileStatus()) {
        std::cerr << "Shader compilation error:" << shader.getInfoLog() << std::endl;
        throw std::runtime_error("Shader compilation error:" + shader.getInfoLog());
    }
    return shader;
}

void loadShaders(const fs::path& directoryPath, ShaderLibrary& shaderLibrary)
{
    for (auto& entry : fs::recursive_directory_iterator(directoryPath)) {
        auto path = entry.path();
        if (fs::is_regular_file(path) && path.extension() == ".glsl") {
            shaderLibrary.emplace(std::make_pair(path, loadShader(path)));
        }
    }
}

c2ba::GLProgram compileProgram(const ShaderLibrary& shaderLibrary, std::vector<fs::path> shaderPaths)
{
    c2ba::GLProgram program;
    for (const auto& path : shaderPaths) {
        program.attachShader(shaderLibrary.at(path));
    }
    program.link();
    if (!program.getLinkStatus()) {
        std::cerr << "Program link error:" << program.getInfoLog() << std::endl;
        throw std::runtime_error("Program link error:" + program.getInfoLog());
    }
    return program;
}

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

    ShaderLibrary shaderLibrary;
    auto shadersRoot = appDir / "glsl";
    loadShaders(shadersRoot, shaderLibrary);

    auto program = compileProgram(shaderLibrary, { shadersRoot / "triangle.vs.glsl", shadersRoot / "triangle.fs.glsl" });

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