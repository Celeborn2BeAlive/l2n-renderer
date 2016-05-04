#version 430

layout(local_size_x = 32, local_size_y = 32) in;

struct PhongMaterial {
    vec4 diffuse;
    vec3 glossy;
    float shininess;
};

struct Sphere {
    vec3 position;
    float radius;
    uint materialID;
    uvec3 align0;
};

struct PointLight {
    vec3 position;
    float align0;
    vec3 radiantIntensity;
    float align1;
};

struct DirectionalLight {
    vec3 incidentDirection;
    float align0;
    vec3 emittedRadiance;
    float align1;
};

uniform uint uIterationCount;
uniform mat4 uViewProjectionMatrix;

layout(std430) buffer MaterialBuffer {
    uint materialCount;
    PhongMaterial materialArray[];
};

layout(std430) buffer SphereBuffer {
    uint sphereCount;
    Sphere sphereArray[];
};

layout(std430) buffer PointLightBuffer {
    uint pointLightCount;
    PointLight pointLightArray[];
};

layout(std430) buffer DirectionalLightBuffer {
    uint directionalLightCount;
    DirectionalLight directionalLightArray[];
};

writeonly uniform image2D uOutputImage;

void main() {
    ivec2 framebufferSize = imageSize(uOutputImage);
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);

    if (pixelCoords.x >= framebufferSize.x || pixelCoords.y >= framebufferSize.y) {
        return;
    }

    float radius = abs(cos(0.01 * float(uIterationCount)));
    vec2 normalizedPosition = 2.0 * ((vec2(pixelCoords) + vec2(0.5)) / vec2(framebufferSize) - vec2(0.5));
    float sqrDist = 2 * abs(radius * radius - dot(normalizedPosition, normalizedPosition));

    vec2 uv = vec2(pixelCoords) / vec2(framebufferSize);
    imageStore(uOutputImage, pixelCoords, vec4(abs(sin(0.02 * float(uIterationCount))), uv * sqrDist, 0));
}