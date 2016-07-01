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
#include <c2ba/glutils/GLNVShaderBufferLoad.hpp>

#include <c2ba/maths/types.hpp>
#include <c2ba/maths/geometry.hpp>
#include <c2ba/maths/sampling/Random.hpp>

#include <imgui/imgui.h>
#include <imgui/examples/opengl3_example/imgui_impl_glfw_gl3.h>

#include "json.hpp"

#include <filesystem>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <fstream>

#include "shaders.hpp"
#include "ViewController.hpp"

#include "tinymt32.hpp"
#include "tinymt32dc.hpp"
#include "shaders.hpp"

using namespace c2ba;
using json = nlohmann::json;

namespace l2n
{

#ifndef _WIN32
#define sprintf_s snprintf
#endif

void formatDebugOutput(char outStr[], size_t outStrSize, GLenum source, GLenum type,
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

    ~Application();

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

    m_pWindow = glfwCreateWindow(int(m_nWindowWidth), int(m_nWindowHeight), "Les Lumieres de Noel", NULL, NULL);
    if (!m_pWindow) {
        std::cerr << "Unable to open window.\n";
        glfwTerminate();
        throw std::runtime_error("Unable to open window.\n");
    }

    glfwMakeContextCurrent(m_pWindow);

    glfwSwapInterval(0);

    if (!gladLoadGL()) {
        std::cerr << "Unable to init OpenGL.\n";
        throw std::runtime_error("Unable to init OpenGL.\n");
    }

    glDebugMessageCallback((GLDEBUGPROCARB)debugCallback, nullptr);
    //glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    m_ShadersRootPath = appDir / "glsl";
    loadShaders(m_ShadersRootPath, m_ShaderLibrary);

    // Setup ImGui binding
    ImGui_ImplGlfwGL3_Init(m_pWindow, true);
}

Application::~Application()
{
    ImGui_ImplGlfwGL3_Shutdown();
    glfwTerminate();
}

struct Sphere {
    float3 center;
    float sqrRadius;
    uint32_t materialID;
    uint32_t padding[3];

    Sphere(const float3& center, float radius) :
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
    //    RandomGenerator rng;
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

    float4x4 uRcpViewMatrix;
    float4x4 uRcpProjMatrix;
    float4x4 uRcpViewProjMatrix;
    float3 uCameraPosition;
    const Sphere* uSphereArray;
    size_t uSphereCount;

    // zAxis should be normalized
    float3x3 frameZ(float3 zAxis)
    {
        float3x3 matrix;

        matrix[2] = zAxis;
        if (abs(zAxis.y) > abs(zAxis.x)) {
            float rcpLength = 1.f / length(float2(zAxis.x, zAxis.y));
            matrix[0] = rcpLength * float3(zAxis.y, -zAxis.x, 0.f);
        }
        else {
            float rcpLength = 1.f / length(float2(zAxis.x, zAxis.z));
            matrix[0] = rcpLength * float3(zAxis.z, 0.f, -zAxis.x);
        }
        matrix[1] = cross(zAxis, matrix[0]);
        return matrix;
    }

    float3 uniformSampleHemisphere(float u1, float u2, float& jacobian)
    {
        float r = sqrt(1.f - u1 * u1);
        float phi = 2.f * pi<float>() * u2;
        jacobian = 2.f * pi<float>();
        return float3(cos(phi) * r, sin(phi) * r, u1);
    }

    float3 cosineSampleHemisphere(float u1, float u2, float& jacobian)
    {
        float r = sqrt(u1);
        float phi = 2.f * pi<float>() * u2;
        float cosTheta = sqrt(max(0.f, 1.f - u1));
        const float x = r * cos(phi);
        const float y = r * sin(phi);
        jacobian = cosTheta > 0.0 ? pi<float>() / cosTheta : cosTheta;
        return float3(x, y, cosTheta);
    }

    float2 uniformSampleDisk(float radius, float u1, float u2, float& jacobian) {
        float r = sqrt(u1);
        float theta = 2 * pi<float>() * u2;
        jacobian = pi<float>() * r * r;
        return radius * r * float2(cos(theta), sin(theta));
    }

    float intersectSphere(float3 org, float3 dir, Sphere sphere, float3& position, float3& normal) {
        float3 centerOrg = org - sphere.center;
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

    float intersectScene(float3 org, float3 dir, float3& position, float3& normal) {
        float currentDist = -1;
        for (auto i = 0; i < uSphereCount; ++i) {
            float3 tmpPos, tmpNormal;
            float t = intersectSphere(org, dir, uSphereArray[i], tmpPos, tmpNormal);
            if (t >= 0.f && (currentDist < 0.f || t < currentDist)) {
                currentDist = t;
                position = tmpPos;
                normal = tmpNormal;
            }
        }
        return currentDist;
    }

    float intersectScene(float3 org, float3 dir, float3& position, float3& normal, uint32_t& sphereIndex) {
        float currentDist = -1;
        for (auto i = 0; i < uSphereCount; ++i) {
            float3 tmpPos, tmpNormal;
            float t = intersectSphere(org, dir, uSphereArray[i], tmpPos, tmpNormal);
            if (t >= 0.f && (currentDist < 0.f || t < currentDist)) {
                currentDist = t;
                position = tmpPos;
                normal = tmpNormal;
                sphereIndex = i;
            }
        }
        return currentDist;
    }

    float3 ambientOcclusion(float3 org, float3 dir, Random& random) {
        float3 position, normal;
        float3 color = float3(0);
        float dist = intersectScene(org, dir, position, normal);
        if (dist >= 0.) {
            //color = normal;

            float3x3 localToWorld = frameZ(normal);
            org = org + dist * dir;
            float2 uv = float2(randFloat(random), randFloat(random));
            float jacobian;
            float3 dir = cosineSampleHemisphere(uv.x, uv.y, jacobian);
            float cosTheta = dir.z;
            dir = localToWorld * dir;
            dist = intersectScene(org + 0.01f * dir, dir, position, normal);
            if (dist < 0.) {
                return float3(/*jacobian * cosTheta / pi<float>()*/1); // Works thanks to importance sampling
            }
        }
        return float3(0.);
    }

    float3 hit(float3 org, float3 dir) {
        float3 position, normal;
        float3 color = float3(0);
        float dist = intersectScene(org, dir, position, normal);
        if (dist >= 0.) {
            return float3(1);
        }
        return float3(0);
    }

    float3 normal(float3 org, float3 dir) {
        float3 position, normal;
        float3 color = float3(0);
        float dist = intersectScene(org, dir, position, normal);
        if (dist >= 0.) {
            return normal;
        }
        return float3(0);
    }

    float3 getColor(uint32_t n) {
        return fract(
            sin(
                float(n + 1) * float3(12.9898, 78.233, 56.128)
                )
            * 43758.5453f
            );
    }

    float luminance(const float3 color) {
        return 0.212671 * color.r + 0.715160 * color.g + 0.072169 * color.b;
    }

    float3 pathtracing(float3 org, float3 dir, tinymt32_t * random)
    {
        float3 sunDirection = normalize(float3(1, 1, -1));

        float3 position, normal;
        float3 throughput = float3(1);
        float3 color = float3(0);
        uint32_t sphereIndex;
        float dist = intersectScene(org, dir, position, normal, sphereIndex);
        uint32_t pathLength = 0;
        while (dist >= 0.0 && pathLength <= 1) {
            ++pathLength;
            float3 Kd = getColor(sphereIndex);

            // One sphere on 16 is emissive
            if (sphereIndex % 16 == 0) {
                float sqrRadius = uSphereArray[sphereIndex].sqrRadius;

                float emissionScale = 8192.;
                color += throughput * emissionScale / (4 * pi<float>() * sqrRadius);
                dist = -1; // Emissive spheres are not reflective
            }
            else {
                float3x3 localToWorld = frameZ(normal);
                org = org + dist * dir;
                float2 uv = float2(tinymt32_generate_floatOO(random), tinymt32_generate_floatOO(random));
                float jacobian;
                float3 localDir = cosineSampleHemisphere(uv.x, uv.y, jacobian);
                float cosTheta = localDir.z;
                dir = localToWorld * localDir;

                throughput *= Kd; // Works thanks to importance sampling, for diffuse spheres

                float rr = tinymt32_generate_floatOO(random);
                float rrProb = min(0.9f, luminance(throughput));
                if (rr < rrProb) {
                    dist = intersectScene(org + 0.01f * dir, dir, position, normal, sphereIndex);
                    throughput /= rrProb;
                }
                else {
                    dist = -1.0;
                }
            }
        }
        // Environment lighting
        if (pathLength == 0 || sphereIndex % 16 != 0)
            color += throughput * 3.f * pow(max(0.f, dot(sunDirection, dir)), 128);

        return color;
    }

    void renderPixels(
        float4* finalImage, float4* accumImage, size_t imageWidth, size_t imageHeight, State& state)
    {
        const auto pixelCount = imageWidth * imageHeight;

        if (state.randoms.size() != pixelCount) {
            RandomGenerator rng;
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

                int2 pixelCoords(pixelIndex % imageWidth, pixelIndex / imageWidth);

                float2 pixelSample = float2(randFloat(*random), randFloat(*random));
                //float pixelSampleJacobian;
                //float2 diskSample = uniformSampleDisk(1, pixelSample.x, pixelSample.y, pixelSampleJacobian);

                float2 rasterCoords = float2(pixelCoords) + pixelSample;
                float2 sampleCoords = rasterCoords / float2(imageWidth, imageHeight);

                float4 ndCoords = float4(-1, -1, -1, 1) + float4(2.f * sampleCoords, 0, 0); // Normalized device coordinates
                float4 viewCoords = uRcpViewProjMatrix * ndCoords;
                viewCoords /= viewCoords.w;

                float3 dir = normalize(float3(viewCoords) - uCameraPosition);
                float3 org = uCameraPosition;

                float3 color = normal(org, dir);
                //float3 color = ambientOcclusion(org, dir, random);
                //vec3 color = vec3(getRandFloat(pixelCoords));

                accumImage[pixelIndex] += float4(color, 1.f);

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
        float4* finalImage, float4* accumImage, size_t imageWidth, size_t imageHeight, tinymt32_t * uRandomStateArray, const int2* uTileArray, size_t tileCountPerIt, size_t uTileOffset)
    {
        const auto tileSize = 16;
        const auto tileCountX = imageWidth / tileSize + ((imageWidth % tileSize > 0) ? 1 : 0);
        const auto tileCountY = imageHeight / tileSize + ((imageHeight % tileSize > 0) ? 1 : 0);
        const auto uTileCount = tileCountX * tileCountY;

        const auto threadCount = std::thread::hardware_concurrency();
        const auto pixelCount = imageWidth * imageHeight;
        std::atomic_size_t nextTileIndex{ 0 };

        const auto jitterSize = 4;
        const auto jitterLength = 1.f / jitterSize;

        auto taskFunctor = [&]()
        {
            size_t tileIndex;
            while (tileCountPerIt > (tileIndex = nextTileIndex++)) {
                
                const int2 tile = uTileArray[(tileIndex + uTileOffset) % uTileCount];
                const size_t tileX = tile.x;
                const size_t tileY = tile.y;

                for (auto pixelY = tileY * tileSize, endY = min(pixelY + tileSize, imageHeight); pixelY < endY; ++pixelY) {
                    for (auto pixelX = tileX * tileSize, endX = min(pixelX + tileSize, imageWidth); pixelX < endX; ++pixelX) {
                        const auto pixelIndex = pixelX + pixelY * imageWidth;

                        auto* const random = uRandomStateArray + pixelIndex;

                        const int2 pixelCoords(pixelIndex % imageWidth, pixelIndex / imageWidth);

                        //const auto jitterIndex = state.iteration % (jitterSize * jitterSize);
                        //const auto jitterX = jitterIndex % jitterSize;
                        //const auto jitterY = jitterIndex / jitterSize;

                        //const float2 jitterSample = float2(0.5);
                        //const float2 pixelSample = jitterLength * float2(jitterX, jitterY) + jitterSample * jitterLength;//float2(randFloat(*random), randFloat(*random));
                        //float pixelSampleJacobian;
                        //float2 diskSample = uniformSampleDisk(1, pixelSample.x, pixelSample.y, pixelSampleJacobian);

                        const float2 pixelSample = float2(tinymt32_generate_floatOO(random), tinymt32_generate_floatOO(random));

                        const float2 rasterCoords = float2(pixelCoords) + pixelSample;
                        const float2 sampleCoords = rasterCoords / float2(imageWidth, imageHeight);

                        const float4 ndCoords = float4(-1, -1, 1, 1) + float4(2.f * sampleCoords, 0, 0); // Normalized device coordinates on the far plane z=1 (seems to be more precise)
                        float4 viewCoords = uRcpViewProjMatrix * ndCoords;
                        viewCoords /= viewCoords.w;
                        //float4 viewCoords = uRcpViewMatrix * float4(ndCoords * float4(1.f, imageHeight / float(imageWidth), 1.f, 1.f));

                        const float3 dir = normalize(float3(viewCoords) - uCameraPosition);
                        const float3 org = uCameraPosition;

                        //const float3 color = hit(org, dir);
                        //const float3 color = normal(org, dir);
                        //float3 color = ambientOcclusion(org, dir, *random);
                        float3 color = pathtracing(org, dir, random);

                        accumImage[pixelIndex] += float4(color, 1.f);

                        finalImage[pixelIndex] = accumImage[pixelIndex] / accumImage[pixelIndex].w;
                        finalImage[pixelIndex] = pow(finalImage[pixelIndex], float4(0.45f));
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
    }

    void render(
        float4* finalImage, float4* accumImage, size_t imageWidth, size_t imageHeight, tinymt32_t* uRandomStateArray, const int2* tiles, size_t tileCountPerIt, size_t tileOffset)
    {
        return renderTiles(finalImage, accumImage, imageWidth, imageHeight, uRandomStateArray, tiles, tileCountPerIt, tileOffset);
    }
};

int Application::run()
{
    float4x4 viewMatrix;

    try {
        if (fs::exists(m_AppPath.parent_path() / "l2n_cache.json")) {
            std::ifstream in{ m_AppPath.parent_path() / "l2n_cache.json" };
            json l2n_cache;
            in >> l2n_cache;
            auto it = l2n_cache.find("view_matrix");
            if (it != std::end(l2n_cache)) {
                std::vector<float> values = *it;
                std::copy(begin(values), end(values), value_ptr(viewMatrix));
            }
            else {
                viewMatrix = transpose(float4x4(0.996, 0.015, 0.084, 12.503, 0.005, 0.974, -0.228, 1.748, -0.085, 0.227, 0.970, -325.982, 0.0, 0.0, 0.0, 1.0));
            }
        }
        else {
            viewMatrix = transpose(float4x4(0.996, 0.015, 0.084, 12.503, 0.005, 0.974, -0.228, 1.748, -0.085, 0.227, 0.970, -325.982, 0.0, 0.0, 0.0, 1.0));
        }
    }
    catch (...)
    {
        std::cerr << "Unable to load json settings file" << std::endl;
        viewMatrix = transpose(float4x4(0.996, 0.015, 0.084, 12.503, 0.005, 0.974, -0.228, 1.748, -0.085, 0.227, 0.970, -325.982, 0.0, 0.0, 0.0, 1.0));
    }

    RandomGenerator rng;

    auto worldSize = 1024.f;
    ViewController viewController{ m_pWindow, worldSize / 10.f };
    viewController.setViewMatrix(viewMatrix);

    //viewController.setViewMatrix(transpose(float4x4(0.996, 0.015, 0.084, 12.503, 0.005, 0.974, -0.228, 1.748, -0.085, 0.227, 0.970, -325.982, 0.0, 0.0, 0.0, 1.0)));

    auto program = compileProgram(m_ShadersRootPath, m_ShaderLibrary, "sphere_pathtracing.cs.glsl", "rand_tinymt32.cs.glsl");
    program.use();

    GLUniform<GLSLImage2Df> uOutputImage{ program, "uOutputImage" };
    uOutputImage.set(program, OUTPUT_IMAGE_BINDING);

    GLUniform<GLSLImage2Df> uAccumImage{ program, "uAccumImage" };
    uAccumImage.set(program, ACCUM_IMAGE_BINDING);

    GLUniform<GLuint> uIterationCount{ program, "uIterationCount" };

    GLUniform<GLfloat4x4> uRcpViewProjMatrix{ program, "uRcpViewProjMatrix" };
    C2BA_GLUNIFORM(program, GLfloat4x4, uRcpViewMatrix);
    C2BA_GLUNIFORM(program, float, uProjTanHalfFovy);
    C2BA_GLUNIFORM(program, float, uProjRatio);

    GLUniform<GLfloat3> uCameraPosition{ program, "uCameraPosition" };

    GLUniform<GLuint> uSphereCount{ program, "uSphereCount" };

    C2BA_GLUNIFORM(program, GLuint, uTileCount);
    C2BA_GLUNIFORM(program, GLuint, uTileOffset);
    GLUniform<GLBufferAddress<Sphere>> uSphereArray{ program, "uSphereArray" };

    GLUniform<GLBufferAddress<int2>> uTileArray{ program, "uTileArray" };

    C2BA_GLUNIFORM(program, GLBufferAddress<tinymt32_t>, uRandomStateArray);

    size_t framebufferWidth = m_nWindowWidth;
    size_t framebufferHeight = m_nWindowHeight;

    const auto projRatio = float(framebufferWidth) / framebufferHeight;
    const auto projTanHalfFovy = tan(0.5f * radians(45.f));
    const auto& projMatrix = perspective(radians(45.f), float(framebufferWidth) / framebufferHeight, 0.01f, 100.f);

    uProjRatio.set(projRatio);
    uProjTanHalfFovy.set(projTanHalfFovy);

    const auto tileSize = 32;
    const auto tileCountX = GLint((framebufferWidth / tileSize) + (framebufferWidth % tileSize != 0));
    const auto tileCountY = GLint((framebufferHeight / tileSize) + (framebufferHeight % tileSize != 0));
    const auto tileCount = tileCountX * tileCountY;
    auto tileCountPerIteration = tileCountX;

    uTileCount.set(program, tileCountX * tileCountY);

    auto computeTileVector = [&]()
    {
        std::vector<int2> tileVector;
        for (auto j = 0; j < tileCountY; ++j) {
            for (auto i = 0; i < tileCountX; ++i) {
                tileVector.emplace_back(int2(i, j));
            }
        }
        std::mt19937 rng;
        std::shuffle(begin(tileVector), end(tileVector), rng);

        return tileVector;
    };
    const auto tileVector = computeTileVector();
    auto tileBuffer = makeBufferStorage(tileVector);
    auto tileBufferAddr = getAddress(tileBuffer);
    makeResident(tileBuffer, GL_READ_ONLY);

    uTileArray.set(program, tileBufferAddr);

    GLFramebuffer2D<1, false> framebuffer;
    framebuffer.init(framebufferWidth, framebufferHeight, { GL_RGBA32F }, GL_NEAREST);
    framebuffer.getColorBuffer(0).bindImage(OUTPUT_IMAGE_BINDING, 0, GL_WRITE_ONLY, GL_RGBA32F);
    framebuffer.getColorBuffer(0).clear(0, GL_RGBA, GL_FLOAT, value_ptr(float4(0)));

    auto computeTinyMTStateVector = [&]()
    {
        std::mt19937 rng;
        std::vector<tinymt32_t> states(framebufferWidth * framebufferHeight);
        std::generate(begin(states), end(states), [&]()
        {
            auto seed = rng();
            tinymt32_params params = precomputed_tinymt_params[rng() % precomputed_tinymt_params_count()];
            tinymt32_t random;
            random.mat1 = params.mat1;
            random.mat2 = params.mat2;
            random.tmat = params.tmat;
            tinymt32_init(&random, seed);
            return random;
        });
        return states;
    };
    auto tinyMTStateVector = computeTinyMTStateVector();
    auto tinyMTStateBuffer = makeBufferStorage(tinyMTStateVector);
    makeResident(tinyMTStateBuffer, GL_READ_WRITE);
    uRandomStateArray.set(program, getAddress(tinyMTStateBuffer));

    GLTexture2D accumImage;
    accumImage.setStorage(1, GL_RGBA32F, framebufferWidth, framebufferHeight);
    accumImage.bindImage(ACCUM_IMAGE_BINDING, 0, GL_READ_WRITE, GL_RGBA32F);
    accumImage.clear(0, GL_RGBA, GL_FLOAT, value_ptr(float4(0)));
    
    std::vector<float4> cpuAccumImage(framebufferWidth * framebufferHeight, float4(0));
    std::vector<float4> cpuFinalImage(framebufferWidth * framebufferHeight, float4(0));

    // Draw on screen
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    auto sphereCount = 1024;
    auto computeSpheres = [&]() {
        std::vector<Sphere> spheres;
        for (auto i = 0; i < sphereCount; ++i) {
            spheres.emplace_back(
                float3(-worldSize * 0.5f + worldSize * rng.getFloat(), -worldSize * 0.5f + worldSize * rng.getFloat(), -worldSize * 0.5f + worldSize * rng.getFloat()),
                0.05f * worldSize * rng.getFloat()
                );
        }
        return spheres;
    };

    // #todo improve buffer interface to be able to pass a memory layout instead of a single type
    // Something like GLBuffer<GLBufferLayout<GLuint, Sphere[]>> myBuffer;
    // Only one unspecified sized array allows, at the end of the type list
    auto sphereVector = computeSpheres();
    auto sphereBuffer = makeBufferStorage(sphereVector);
    makeResident(sphereBuffer, GL_READ_ONLY);
    uSphereArray.set(program, getAddress(sphereBuffer));

    CPUSpherePathtracing spherePathracing;
    CPUSpherePathtracing::State pathtracingState;
    spherePathracing.uSphereArray = sphereVector.data();
    spherePathracing.uSphereCount = sphereCount;

    std::cerr << projMatrix << std::endl;
    std::cerr << inverse(projMatrix) << std::endl;

    size_t tileOffset = 0;
    const auto gpuRender = [&](size_t iterationCount)
    {
        uIterationCount.set(iterationCount);
        auto rcpViewProjMatrix = inverse(projMatrix * viewController.getViewMatrix());
        uRcpViewProjMatrix.set(value_ptr(rcpViewProjMatrix));
        uRcpViewMatrix.set(value_ptr(viewController.getRcpViewMatrix()));
        uCameraPosition.set(value_ptr(viewController.getRcpViewMatrix()[3]));
        uSphereCount.set(GLuint(sphereCount));
        uTileOffset.set(tileOffset);

        glDispatchCompute(tileCountPerIteration, 1, 1);

        tileOffset += tileCountPerIteration;
        tileOffset = tileOffset % tileCount;
    };

    const auto cpuRender = [&](size_t iterationCount)
    {
        spherePathracing.uCameraPosition = float3(viewController.getRcpViewMatrix()[3]);
        spherePathracing.uRcpViewMatrix = viewController.getRcpViewMatrix();
        spherePathracing.uRcpViewProjMatrix = inverse(projMatrix * viewController.getViewMatrix());
        spherePathracing.uRcpProjMatrix = inverse(projMatrix);

        spherePathracing.render(cpuFinalImage.data(), cpuAccumImage.data(), framebufferWidth, framebufferHeight, tinyMTStateVector.data(), tileVector.data(), tileCountPerIteration, tileOffset);
        framebuffer.getColorBuffer(0).setSubImage(0, GL_RGBA, GL_FLOAT, cpuFinalImage.data());

        tileOffset += tileCountPerIteration;
        tileOffset = tileOffset % tileCount;
    };


    /* Loop until the user closes the window */
    for (auto iterationCount = 0u; !glfwWindowShouldClose(m_pWindow); ++iterationCount) {
        auto seconds = glfwGetTime();

        gpuRender(iterationCount);
        //cpuRender(iterationCount);

        glClear(GL_COLOR_BUFFER_BIT);

        framebuffer.bindForReading();
        framebuffer.setReadBuffer(0);

        glBlitFramebuffer(
            0, 0, GLint(framebufferWidth), GLint(framebufferHeight),
            0, 0, GLint(m_nWindowWidth), GLint(m_nWindowHeight),
            GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

        ImGui_ImplGlfwGL3_NewFrame();

        {
            ImGui::Begin("Params");
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::InputInt("tileCountPerIteration", &tileCountPerIteration);
            
            ImGui::End();
        }

        // Rendering
        int display_w, display_h;
        glfwGetFramebufferSize(m_pWindow, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImGui::Render();

        /* Poll for and process events */
        glfwPollEvents();

        /* Swap front and back buffers*/
        glfwSwapBuffers(m_pWindow);

        auto ellapsedTime = glfwGetTime() - seconds;
        auto guiHasFocus = ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard;
        if (!guiHasFocus && viewController.update(float(ellapsedTime))) {
            accumImage.clear(0, GL_RGBA, GL_FLOAT, value_ptr(float4(0)));
            std::fill(begin(cpuAccumImage), end(cpuAccumImage), float4(0));
        }
    }

    {
        json l2n_cache;
        std::vector<float> values(16);
        auto ptr = value_ptr(viewController.getViewMatrix());
        std::copy(ptr, ptr + 16, begin(values));
        l2n_cache["view_matrix"] = values;
        std::ofstream out{ m_AppPath.parent_path() / "l2n_cache.json" };
        out << l2n_cache.dump(4);
    }

    return 0;
}

}

int main(int argc, char** argv)
{
    l2n::Application app(argc, argv);
    return app.run();
}