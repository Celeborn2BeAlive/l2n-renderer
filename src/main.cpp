#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glad/glad.h>

#define C2BA_GLUTILS_GL3_HEADER <glad/glad.h>
#include <c2ba/glutils/GLBuffer.hpp>
#include <c2ba/glutils/GLVertexArray.hpp>
#include <c2ba/glutils/GLShader.hpp>
#include <c2ba/glutils/GLProgram.hpp>
#include <c2ba/glutils/GLTexture.hpp>
#include <c2ba/glutils/GLFramebuffer.hpp>
#include <c2ba/glutils/GLUniform.hpp>

#include <c2ba/maths/types.hpp>

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
        { ".fs", { GL_FRAGMENT_SHADER, "fragment" } },
        { ".gs",{ GL_GEOMETRY_SHADER, "geometry" } },
        { ".cs",{ GL_COMPUTE_SHADER, "compute" } }
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

#ifndef _WIN32
#define sprintf_s snprintf
#endif

static void formatDebugOutput(char outStr[], size_t outStrSize, GLenum source, GLenum type,
    GLuint id, GLenum severity, const char *msg)
{
    char sourceStr[32];
    const char *sourceFmt = "UNDEFINED(0x%04X)";
    switch (source)

    {
    case GL_DEBUG_SOURCE_API_ARB:             sourceFmt = "API"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:   sourceFmt = "WINDOW_SYSTEM"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB: sourceFmt = "SHADER_COMPILER"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:     sourceFmt = "THIRD_PARTY"; break;
    case GL_DEBUG_SOURCE_APPLICATION_ARB:     sourceFmt = "APPLICATION"; break;
    case GL_DEBUG_SOURCE_OTHER_ARB:           sourceFmt = "OTHER"; break;
    }

    sprintf_s(sourceStr, 32, sourceFmt, source);

    char typeStr[32];
    const char *typeFmt = "UNDEFINED(0x%04X)";
    switch (type)
    {

    case GL_DEBUG_TYPE_ERROR_ARB:               typeFmt = "ERROR"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB: typeFmt = "DEPRECATED_BEHAVIOR"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:  typeFmt = "UNDEFINED_BEHAVIOR"; break;
    case GL_DEBUG_TYPE_PORTABILITY_ARB:         typeFmt = "PORTABILITY"; break;
    case GL_DEBUG_TYPE_PERFORMANCE_ARB:         typeFmt = "PERFORMANCE"; break;
    case GL_DEBUG_TYPE_OTHER_ARB:               typeFmt = "OTHER"; break;
    }
    sprintf_s(typeStr, 32, typeFmt, type);


    char severityStr[32];
    const char *severityFmt = "UNDEFINED";
    switch (severity)
    {
    case GL_DEBUG_SEVERITY_HIGH_ARB:   severityFmt = "HIGH";   break;
    case GL_DEBUG_SEVERITY_MEDIUM_ARB: severityFmt = "MEDIUM"; break;
    case GL_DEBUG_SEVERITY_LOW_ARB:    severityFmt = "LOW"; break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:    severityFmt = "NOTIFICATION"; break;
    }

    sprintf_s(severityStr, 32, severityFmt, severity);

    sprintf_s(outStr, outStrSize, "OpenGL: %s [source=%s type=%s severity=%s id=%d]",
        msg, sourceStr, typeStr, severityStr, id);
}

static void debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, GLvoid* userParam)
{
    char finalMessage[256];
    formatDebugOutput(finalMessage, 256, source, type, id, severity, message);
    std::cerr << finalMessage << "\n";
}

class Application 
{
public:
    Application(int argc, char** argv);

    int run();
private:
    fs::path m_AppPath;
    fs::path m_ShadersRootPath;

    size_t m_nWindowWidth = 1280;
    size_t m_nWindowHeight = 720;

    GLFWwindow* m_pWindow = nullptr;

    ShaderLibrary m_ShaderLibrary;
};

Application::Application(int argc, char** argv)
{
    m_AppPath = fs::path{ argv[0] };
    auto appDir = m_AppPath.parent_path();

    if (!glfwInit()) {
        std::cerr << "Unable to init GLFW.\n";
        throw std::runtime_error("Unable to init GLFW.\n");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    m_pWindow = glfwCreateWindow(m_nWindowWidth, m_nWindowHeight, "Les lumieres de Noel", NULL, NULL);
    if (!m_pWindow) {
        std::cerr << "Unable to open window.\n";
        glfwTerminate();
        throw std::runtime_error("Unable to open window.\n");
    }

    glfwMakeContextCurrent(m_pWindow);

    if (!gladLoadGL()) {
        std::cerr << "Unable to init OpenGL.\n";
        throw std::runtime_error("Unable to init OpenGL.\n");
    }

    glDebugMessageCallback((GLDEBUGPROCARB)debugCallback, nullptr);
    //glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    m_ShadersRootPath = appDir / "glsl";
    loadShaders(m_ShadersRootPath, m_ShaderLibrary);
}

int Application::run()
{
    auto uvProgram = compileProgram(m_ShaderLibrary, { m_ShadersRootPath / "sphere_pathtracing.cs.glsl" });
    c2ba::GLUniform<c2ba::GLSLImage2Df> uOutputImage{ uvProgram, "uOutputImage" };
    uOutputImage.set(uvProgram, 0);
    c2ba::GLUniform<GLuint> uIterationCount{ uvProgram, "uIterationCount" };

    size_t framebufferWidth = 1024;
    size_t framebufferHeight = 748;

    auto numGroupX = (framebufferWidth / 32) + (framebufferWidth % 32 != 0);
    auto numGroupY = (framebufferHeight / 32) + (framebufferHeight % 32 != 0);

    c2ba::GLFramebuffer2D<1, false> framebuffer;
    framebuffer.init(framebufferWidth, framebufferHeight, { GL_RGBA32F }, GL_NEAREST);
    framebuffer.getColorBuffer(0).bindImage(0, 0, GL_WRITE_ONLY, GL_RGBA32F);

    std::vector<c2ba::float4> pixels(framebufferWidth * framebufferHeight);
    for (auto j = 0u; j < framebufferHeight; ++j) {
        for (auto i = 0u; i < framebufferWidth; ++i) {
            pixels[i + j * framebufferWidth] = c2ba::float4(float(i) / framebufferWidth, float(j) / framebufferHeight, 1, 1);
        }
    }

    framebuffer.getColorBuffer(0).setSubImage(0, GL_RGBA, GL_FLOAT, pixels.data());

    uvProgram.use();

    // Draw on screen
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    /* Loop until the user closes the window */
    for (auto iterationCount = 0; !glfwWindowShouldClose(m_pWindow); ++iterationCount) {
        uIterationCount.set(iterationCount);
        glDispatchCompute(numGroupX, numGroupY, 1);

        glClear(GL_COLOR_BUFFER_BIT);

        framebuffer.bindForReading();
        framebuffer.setReadBuffer(0);

        glBlitFramebuffer(0, 0, framebufferWidth, framebufferHeight,
            0, 0, m_nWindowWidth, m_nWindowHeight,
            GL_COLOR_BUFFER_BIT, GL_LINEAR);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

        /* Swap front and back buffers */
        glfwSwapBuffers(m_pWindow);

        /* Poll for and process events */
        glfwPollEvents();
    }

    return 0;
}

}

int main(int argc, char** argv)
{
    l2n::Application app(argc, argv);
    return app.run();
}