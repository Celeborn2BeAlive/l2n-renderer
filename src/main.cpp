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
#include <c2ba/maths/geometry.hpp>
#include <c2ba/maths/sampling/Random.hpp>

#include <filesystem>
#include <unordered_map>
#include <thread>
#include <atomic>

#include "shaders.hpp"
#include "ViewController.hpp"

#include "tinymt32.hpp"

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
    char finalMessage[2048];
    formatDebugOutput(finalMessage, 2048, source, type, id, severity, message);
    std::cerr << finalMessage << "\n\n";
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

    m_pWindow = glfwCreateWindow(int(m_nWindowWidth), int(m_nWindowHeight), "Les lumieres de Noel", NULL, NULL);
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

struct Sphere {
    c2ba::float3 center;
    float sqrRadius;
    uint32_t materialID;
    c2ba::uint3 align0;

    Sphere(const c2ba::float3& center, float radius) :
        center(center), sqrRadius(radius * radius), materialID(0) {
    }
};

enum ImageBindings 
{
    OUTPUT_IMAGE_BINDING = 0,
    RANDOM_STATE_IMAGE_BINDING = 1,
    ACCUM_IMAGE_BINDING = 2,
    TINYMT_RANDOM_STATE_IMAGE_BINDING = 3,
    TINYMT_RANDOM_MAT_IMAGE_BINDING = 4
};

struct CPUSpherePathtracing
{
    struct Random
    {
        tinymt32_t random;
    };

    static void setSeed(Random& random, uint32_t seed)
    {
        tinymt32_init(&random.random, seed);
    }

    static float randFloat(Random& random)
    {
        return tinymt32_generate_floatOO(&random.random);
    }

    //struct Random
    //{
    //    c2ba::RandomGenerator rng;
    //};

    //static void setSeed(Random& random, uint32_t seed)
    //{
    //    random.rng.setSeed(seed);
    //}

    //static float randFloat(Random& random)
    //{
    //    return random.rng.getFloat();
    //}

    struct State
    {
        std::vector<Random> randoms;
        size_t iteration;
    };

    c2ba::float4x4 uRcpViewMatrix;
    c2ba::float4x4 uRcpViewProjMatrix;
    c2ba::float3 uCameraPosition;
    const Sphere* sphereArray;
    size_t uSphereCount;

    // zAxis should be normalized
    c2ba::float3x3 frameZ(c2ba::float3 zAxis)
    {
        c2ba::float3x3 matrix;

        matrix[2] = zAxis;
        if (abs(zAxis.y) > abs(zAxis.x)) {
            float rcpLength = 1.f / length(c2ba::float2(zAxis.x, zAxis.y));
            matrix[0] = rcpLength * c2ba::float3(zAxis.y, -zAxis.x, 0.f);
        }
        else {
            float rcpLength = 1.f / length(c2ba::float2(zAxis.x, zAxis.z));
            matrix[0] = rcpLength * c2ba::float3(zAxis.z, 0.f, -zAxis.x);
        }
        matrix[1] = cross(zAxis, matrix[0]);
        return matrix;
    }

    c2ba::float3 uniformSampleHemisphere(float u1, float u2, float& jacobian)
    {
        float r = sqrt(1.f - u1 * u1);
        float phi = 2.f * c2ba::pi<float>() * u2;
        jacobian = 2.f * c2ba::pi<float>();
        return c2ba::float3(cos(phi) * r, sin(phi) * r, u1);
    }

    c2ba::float3 cosineSampleHemisphere(float u1, float u2, float& jacobian)
    {
        float r = sqrt(u1);
        float phi = 2.f * c2ba::pi<float>() * u2;
        float cosTheta = sqrt(c2ba::max(0.f, 1.f - u1));
        const float x = r * cos(phi);
        const float y = r * sin(phi);
        jacobian = cosTheta > 0.0 ? c2ba::pi<float>() / cosTheta : cosTheta;
        return c2ba::float3(x, y, cosTheta);
    }

    c2ba::float2 uniformSampleDisk(float radius, float u1, float u2, float& jacobian) {
        float r = sqrt(u1);
        float theta = 2 * c2ba::pi<float>() * u2;
        jacobian = c2ba::pi<float>() * r * r;
        return radius * r * c2ba::float2(cos(theta), sin(theta));
    }

    float intersectSphere(c2ba::float3 org, c2ba::float3 dir, Sphere sphere, c2ba::float3& position, c2ba::float3& normal) {
        c2ba::float3 centerOrg = org - sphere.center;
        float a = 1;
        float b = 2 * dot(centerOrg, dir);
        float c = dot(centerOrg, centerOrg) - sphere.sqrRadius;
        float discriminant = b * b - 4 * c;
        if (discriminant < 0.) {
            return -1.;
        }
        float sqrtDiscriminant = sqrt(discriminant);
        float t1 = 0.5f * (-b - sqrtDiscriminant);
        float t2 = 0.5f * (-b + sqrtDiscriminant);
        float t = (t1 >= 0.f) ? t1 : t2;

        position = org + t * dir;
        normal = normalize(position - sphere.center);

        return t;
    }

    float intersectScene(c2ba::float3 org, c2ba::float3 dir, c2ba::float3& position, c2ba::float3& normal) {
        float currentDist = -1;
        for (auto i = 0; i < uSphereCount; ++i) {
            c2ba::float3 tmpPos, tmpNormal;
            float t = intersectSphere(org, dir, sphereArray[i], tmpPos, tmpNormal);
            if (t >= 0.f && (currentDist < 0.f || t < currentDist)) {
                currentDist = t;
                position = tmpPos;
                normal = tmpNormal;
            }
        }
        return currentDist;
    }

    c2ba::float3 ambientOcclusion(c2ba::float3 org, c2ba::float3 dir, Random& random) {
        c2ba::float3 position, normal;
        c2ba::float3 color = c2ba::float3(0);
        float dist = intersectScene(org, dir, position, normal);
        if (dist >= 0.) {
            //color = normal;

            c2ba::float3x3 localToWorld = frameZ(normal);
            org = org + dist * dir;
            c2ba::float2 uv = c2ba::float2(randFloat(random), randFloat(random));
            float jacobian;
            c2ba::float3 dir = cosineSampleHemisphere(uv.x, uv.y, jacobian);
            float cosTheta = dir.z;
            dir = localToWorld * dir;
            dist = intersectScene(org + 0.01f * dir, dir, position, normal);
            if (dist < 0.) {
                return c2ba::float3(/*jacobian * cosTheta / c2ba::pi<float>()*/1); // Works thanks to importance sampling
            }
        }
        return c2ba::float3(0.);
    }

    c2ba::float3 hit(c2ba::float3 org, c2ba::float3 dir) {
        c2ba::float3 position, normal;
        c2ba::float3 color = c2ba::float3(0);
        float dist = intersectScene(org, dir, position, normal);
        if (dist >= 0.) {
            return c2ba::float3(1);
        }
        return c2ba::float3(0);
    }

    c2ba::float3 normal(c2ba::float3 org, c2ba::float3 dir) {
        c2ba::float3 position, normal;
        c2ba::float3 color = c2ba::float3(0);
        float dist = intersectScene(org, dir, position, normal);
        if (dist >= 0.) {
            return normal;
        }
        return c2ba::float3(0);
    }

    void renderPixels(
        c2ba::float4* finalImage, c2ba::float4* accumImage, size_t imageWidth, size_t imageHeight, State& state)
    {
        const auto pixelCount = imageWidth * imageHeight;

        if (state.randoms.size() != pixelCount) {
            c2ba::RandomGenerator rng;
            std::vector<Random> randomsVector(pixelCount);
            for (auto i = 0u; i < pixelCount; ++i) {
                auto seed = rng.getUInt();
                setSeed(randomsVector[i], seed);
            }
            std::swap(randomsVector, state.randoms);

            state.iteration = 0;
        }
        
        auto* randoms = state.randoms.data();

        const auto threadCount = std::thread::hardware_concurrency();
        std::atomic_size_t nextPixelIndex{ 0 };

        auto taskFunctor = [&]()
        {
            size_t pixelIndex;
            while (pixelCount > (pixelIndex = nextPixelIndex++)) {
                auto* random = randoms + pixelIndex;

                c2ba::int2 pixelCoords(pixelIndex % imageWidth, pixelIndex / imageWidth);

                c2ba::float2 pixelSample = c2ba::float2(randFloat(*random), randFloat(*random));
                //float pixelSampleJacobian;
                //c2ba::float2 diskSample = uniformSampleDisk(1, pixelSample.x, pixelSample.y, pixelSampleJacobian);

                c2ba::float2 rasterCoords = c2ba::float2(pixelCoords) + pixelSample;
                c2ba::float2 sampleCoords = rasterCoords / c2ba::float2(imageWidth, imageHeight);

                c2ba::float4 ndCoords = c2ba::float4(-1, -1, -1, 1) + c2ba::float4(2.f * sampleCoords, 0, 0); // Normalized device coordinates
                c2ba::float4 viewCoords = uRcpViewProjMatrix * ndCoords;
                viewCoords /= viewCoords.w;

                c2ba::float3 dir = normalize(c2ba::float3(viewCoords) - uCameraPosition);
                c2ba::float3 org = uCameraPosition;

                c2ba::float3 color = normal(org, dir);
                //c2ba::float3 color = ambientOcclusion(org, dir, random);
                //vec3 color = vec3(getRandFloat(pixelCoords));

                accumImage[pixelIndex] += c2ba::float4(color, 1.f);

                finalImage[pixelIndex] = accumImage[pixelIndex] / accumImage[pixelIndex].w;
            }
        };

        std::vector<std::thread> threads;
        for (auto i = 0u; i < threadCount; ++i) {
            threads.emplace_back(taskFunctor);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        ++state.iteration;
    }

    void renderTiles(
        c2ba::float4* finalImage, c2ba::float4* accumImage, size_t imageWidth, size_t imageHeight, State& state)
    {
        const auto tileSize = 32;
        const auto tileCountX = imageWidth / tileSize + ((imageWidth % tileSize > 0) ? 1 : 0);
        const auto tileCountY = imageHeight / tileSize + ((imageHeight % tileSize > 0) ? 1 : 0);
        const auto tileCount = tileCountX * tileCountY;

        if (state.randoms.size() != tileCount) {
            c2ba::RandomGenerator rng;
            std::vector<Random> randomsVector(tileCount);
            for (auto i = 0u; i < tileCount; ++i) {
                auto seed = rng.getUInt();
                setSeed(randomsVector[i], seed);
            }
            std::swap(randomsVector, state.randoms);

            state.iteration = 0;
        }

        auto* randoms = state.randoms.data();

        const auto threadCount = std::thread::hardware_concurrency();
        const auto pixelCount = imageWidth * imageHeight;
        std::atomic_size_t nextTileIndex{ 0 };

        const auto jitterSize = 2;
        const auto jitterLength = 0.5f / jitterSize;

        auto taskFunctor = [&]()
        {
            size_t tileIndex;
            while (tileCount > (tileIndex = nextTileIndex++)) {
                const auto tileX = tileIndex % tileCountX;
                const auto tileY = tileIndex / tileCountX;

                auto* const random = randoms + tileIndex;

                for (auto pixelY = tileY * tileSize, endY = c2ba::min(pixelY + tileSize, imageHeight); pixelY < endY; ++pixelY) {
                    for (auto pixelX = tileX * tileSize, endX = c2ba::min(pixelX + tileSize, imageWidth); pixelX < endX; ++pixelX) {
                        const auto pixelIndex = pixelX + pixelY * imageWidth;

                        const c2ba::int2 pixelCoords(pixelIndex % imageWidth, pixelIndex / imageWidth);

                        const auto jitterX = state.iteration % jitterSize;
                        const auto jitterY = state.iteration / jitterSize;

                        const  c2ba::float2 pixelSample = jitterLength * c2ba::float2(jitterX, jitterY);//c2ba::float2(randFloat(*random), randFloat(*random));
                        //float pixelSampleJacobian;
                        //c2ba::float2 diskSample = uniformSampleDisk(1, pixelSample.x, pixelSample.y, pixelSampleJacobian);

                        const c2ba::float2 rasterCoords = c2ba::float2(pixelCoords) + pixelSample;
                        const c2ba::float2 sampleCoords = rasterCoords / c2ba::float2(imageWidth, imageHeight);

                        const c2ba::float4 ndCoords = c2ba::float4(-1, -1, -1, 1) + c2ba::float4(2.f * sampleCoords, 0, 0); // Normalized device coordinates
                        c2ba::float4 viewCoords = uRcpViewProjMatrix * ndCoords;
                        viewCoords /= viewCoords.w;

                        const c2ba::float3 dir = normalize(c2ba::float3(viewCoords) - uCameraPosition);
                        const c2ba::float3 org = uCameraPosition;

                        const c2ba::float3 color = hit(org, dir);
                        //const c2ba::float3 color = normal(org, dir);
                        //c2ba::float3 color = ambientOcclusion(org, dir, random);
                        //vec3 color = vec3(getRandFloat(pixelCoords));

                        accumImage[pixelIndex] += c2ba::float4(color, 1.f);

                        finalImage[pixelIndex] = accumImage[pixelIndex] / accumImage[pixelIndex].w;
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        for (auto i = 0u; i < threadCount; ++i) {
            threads.emplace_back(taskFunctor);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        ++state.iteration;
    }

    void render(
        c2ba::float4* finalImage, c2ba::float4* accumImage, size_t imageWidth, size_t imageHeight, State& state)
    {
        return renderTiles(finalImage, accumImage, imageWidth, imageHeight, state);
    }
};

int Application::run()
{
    c2ba::RandomGenerator rng;

    auto worldSize = 256.f;
    ViewController viewController{ m_pWindow, worldSize / 10.f };
    viewController.setViewMatrix(c2ba::transpose(c2ba::float4x4(0.996, 0.015, 0.084, 12.503, 0.005, 0.974, -0.228, 1.748, -0.085, 0.227, 0.970, -325.982, 0.0, 0.0, 0.0, 1.0)));

    auto program = compileProgram(m_ShaderLibrary, { m_ShadersRootPath / "sphere_pathtracing.cs.glsl" });
    program.use();

    c2ba::GLUniform<c2ba::GLSLImage2Df> uOutputImage{ program, "uOutputImage" };
    uOutputImage.set(program, OUTPUT_IMAGE_BINDING);

    c2ba::GLUniform<c2ba::GLSLImage2Df> uRandomStateImage{ program, "uRandomStateImage" };
    uRandomStateImage.set(program, RANDOM_STATE_IMAGE_BINDING);

    c2ba::GLUniform<c2ba::GLSLImage2Df> uTinyMTRandomStateImage{ program, "uTinyMTRandomStateImage" };
    uTinyMTRandomStateImage.set(program, TINYMT_RANDOM_STATE_IMAGE_BINDING);

    c2ba::GLUniform<c2ba::GLSLImage2Df> uTinyMTRandomMatImage{ program, "uTinyMTRandomMatImage" };
    uTinyMTRandomMatImage.set(program, TINYMT_RANDOM_MAT_IMAGE_BINDING);

    c2ba::GLUniform<c2ba::GLSLImage2Df> uAccumImage{ program, "uAccumImage" };
    uAccumImage.set(program, ACCUM_IMAGE_BINDING);

    c2ba::GLUniform<GLuint> uIterationCount{ program, "uIterationCount" };

    c2ba::GLUniform<c2ba::GLfloat4x4> uRcpViewProjMatrix{ program, "uRcpViewProjMatrix" };

    c2ba::GLUniform<c2ba::GLfloat3> uCameraPosition{ program, "uCameraPosition" };

    c2ba::GLUniform<GLuint> uSphereCount{ program, "uSphereCount" };

    size_t framebufferWidth = m_nWindowWidth;
    size_t framebufferHeight = m_nWindowHeight;

    const auto& projMatrix = c2ba::perspective(45.f, float(framebufferWidth) / framebufferHeight, 0.01f, 100.f);

    auto numGroupX = GLuint((framebufferWidth / 32) + (framebufferWidth % 32 != 0));
    auto numGroupY = GLuint((framebufferHeight / 32) + (framebufferHeight % 32 != 0));

    c2ba::GLFramebuffer2D<1, false> framebuffer;
    framebuffer.init(framebufferWidth, framebufferHeight, { GL_RGBA32F }, GL_NEAREST);
    framebuffer.getColorBuffer(0).bindImage(OUTPUT_IMAGE_BINDING, 0, GL_WRITE_ONLY, GL_RGBA32F);
    framebuffer.getColorBuffer(0).clear(0, GL_RGBA, GL_FLOAT, c2ba::value_ptr(c2ba::float4(0)));

    c2ba::GLTexture2D randomImage;
    randomImage.setStorage(1, GL_RGBA32UI, framebufferWidth, framebufferHeight);
    randomImage.bindImage(RANDOM_STATE_IMAGE_BINDING, 0, GL_READ_WRITE, GL_RGBA32UI);
    {
        std::vector<c2ba::uint4> pixels(framebufferWidth * framebufferHeight);
        std::generate(begin(pixels), end(pixels), [&]() {
            auto randomState = c2ba::uint4(0);
            for (auto i = 0; i < 4; ++i) {
                while (randomState[i] <= 128) {
                    randomState[i] = rng.getUInt();
                }
            }
            return randomState;
        });
        randomImage.setSubImage(0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, pixels.data());
    }

    c2ba::GLTexture2D tinymt32StateImage;
    c2ba::GLTexture2D tinymt32MatImage;
    tinymt32StateImage.setStorage(1, GL_RGBA32UI, framebufferWidth, framebufferHeight);
    tinymt32StateImage.bindImage(TINYMT_RANDOM_STATE_IMAGE_BINDING, 0, GL_READ_WRITE, GL_RGBA32UI);
    tinymt32MatImage.setStorage(1, GL_RGBA32UI, framebufferWidth, framebufferHeight);
    tinymt32MatImage.bindImage(TINYMT_RANDOM_MAT_IMAGE_BINDING, 0, GL_READ_WRITE, GL_RGBA32UI);
    {
        std::vector<c2ba::uint4> states(framebufferWidth * framebufferHeight);
        std::vector<c2ba::uint4> mats(framebufferWidth * framebufferHeight);

        for (auto i = 0u; i < framebufferWidth * framebufferHeight; ++i) {
            auto seed = rng.getUInt();
            tinymt32_t random;
            tinymt32_init(&random, seed);
            states[i] = c2ba::uint4(random.status[0], random.status[1], random.status[2], random.status[3]);
            mats[i] = c2ba::uint4(random.mat1, random.mat2, random.tmat, 0);
        }

        tinymt32StateImage.setSubImage(0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, states.data());
        tinymt32MatImage.setSubImage(0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, mats.data());
    }

    c2ba::GLTexture2D accumImage;
    accumImage.setStorage(1, GL_RGBA32F, framebufferWidth, framebufferHeight);
    accumImage.bindImage(ACCUM_IMAGE_BINDING, 0, GL_READ_WRITE, GL_RGBA32F);
    accumImage.clear(0, GL_RGBA, GL_FLOAT, c2ba::value_ptr(c2ba::float4(0)));
    
    std::vector<c2ba::float4> cpuAccumImage(framebufferWidth * framebufferHeight, c2ba::float4(0));
    std::vector<c2ba::float4> cpuFinalImage(framebufferWidth * framebufferHeight, c2ba::float4(0));

    // Draw on screen
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    auto sphereCount = 128;
    auto computeSpheres = [&]() {
        std::vector<Sphere> spheres;
        for (auto i = 0; i < sphereCount; ++i) {
            spheres.emplace_back(
                c2ba::float3(-worldSize * 0.5f + worldSize * rng.getFloat(), -worldSize * 0.5f + worldSize * rng.getFloat(), -worldSize * 0.5f + worldSize * rng.getFloat()),
                0.1f * worldSize * rng.getFloat()
                );
        }
        return spheres;
    };

    // #todo improve buffer interface to be able to pass a memory layout instead of a single type
    // Something like GLBuffer<GLBufferLayout<GLuint, Sphere[]>> myBuffer;
    // Only one unspecified sized array allows, at the end of the type list
    auto sphereVector = computeSpheres();
    auto sphereBuffer = c2ba::genBufferStorage<Sphere>(sphereCount, sphereVector.data(), 0);
    sphereBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

    CPUSpherePathtracing spherePathracing;
    CPUSpherePathtracing::State pathtracingState;
    spherePathracing.sphereArray = sphereVector.data();
    spherePathracing.uSphereCount = sphereVector.size();

    /* Loop until the user closes the window */
    for (auto iterationCount = 0; !glfwWindowShouldClose(m_pWindow); ++iterationCount) {
        auto seconds = glfwGetTime();

        /*uIterationCount.set(iterationCount);
        auto rcpViewProjMatrix = c2ba::inverse(projMatrix * viewController.getViewMatrix());
        uRcpViewProjMatrix.set(c2ba::value_ptr(rcpViewProjMatrix));
        uCameraPosition.set(c2ba::value_ptr(viewController.getRcpViewMatrix()[3]));
        uSphereCount.set(GLuint(sphereBuffer.size()));

        glDispatchCompute(numGroupX, numGroupY, 1);*/


        spherePathracing.uCameraPosition = c2ba::float3(viewController.getRcpViewMatrix()[3]);
        spherePathracing.uRcpViewProjMatrix = c2ba::inverse(projMatrix * viewController.getViewMatrix());

        spherePathracing.render(cpuFinalImage.data(), cpuAccumImage.data(), framebufferWidth, framebufferHeight, pathtracingState);
        framebuffer.getColorBuffer(0).setSubImage(0, GL_RGBA, GL_FLOAT, cpuFinalImage.data());

        glClear(GL_COLOR_BUFFER_BIT);

        framebuffer.bindForReading();
        framebuffer.setReadBuffer(0);

        glBlitFramebuffer(
            0, 0, GLint(framebufferWidth), GLint(framebufferHeight),
            0, 0, GLint(m_nWindowWidth), GLint(m_nWindowHeight),
            GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

        /* Poll for and process events */
        glfwPollEvents();

        /* Swap front and back buffers*/
        glfwSwapBuffers(m_pWindow);

        auto ellapsedTime = glfwGetTime() - seconds;
        if (viewController.update(float(ellapsedTime))) {
            accumImage.clear(0, GL_RGBA, GL_FLOAT, c2ba::value_ptr(c2ba::float4(0)));
            std::fill(begin(cpuAccumImage), end(cpuAccumImage), c2ba::float4(0));
        }

        //std::cerr << viewController.getViewMatrix() << std::endl;
    }

    return 0;
}

}

int main(int argc, char** argv)
{
    l2n::Application app(argc, argv);
    return app.run();
}