#version 430

layout(local_size_x = 32, local_size_y = 32) in;

#define M_PI 3.14159265358979323846

struct PhongMaterial {
    vec4 diffuse;
    vec3 glossy;
    float shininess;
};

struct Sphere {
    vec3 center;
    float sqrRadius;
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
uniform mat4 uRcpViewProjMatrix; // Transform normalized device coordinates to world space
uniform vec3 uCameraPosition;


uniform uint uMaterialCount;

layout(std430, binding = 0) buffer MaterialBuffer {
    PhongMaterial materialArray[];
};

uniform uint uSphereCount;

layout(std430, binding = 1) buffer SphereBuffer {
    Sphere sphereArray[];
};

uniform uint uPointLightCount;

layout(std430, binding = 2) buffer PointLightBuffer {
    PointLight pointLightArray[];
};

uniform uint uDirectionalLightCount;

layout(std430, binding = 3) buffer DirectionalLightBuffer {
    DirectionalLight directionalLightArray[];
};

layout(rgba32ui) uniform uimage2D uRandomStateImage;
layout(rgba32f) uniform image2D uAccumImage;

writeonly uniform image2D uOutputImage;

uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
    uint b = (((z << S1) ^ z) >> S2);
    return (((z & M) << S3) ^ b);    
}

uint LCGStep(uint z, uint A, uint C)
{
    return (A * z + C);    
}

float stateToFloat(inout uvec4 state) {
    state.x = TausStep(state.x, 13, 19, 12, 4294967294);
    state.y = TausStep(state.y, 2, 25, 4, 4294967288);
    state.z = TausStep(state.z, 3, 11, 17, 4294967280);
    state.w = LCGStep(state.w, 1664525, 1013904223);

    return 2.3283064365387e-10 * float(state.x ^ state.y ^ state.z ^ state.w);
}

// Hybrid Generator described in  http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
// More details here: http://math.stackexchange.com/questions/337782/pseudo-random-number-generation-on-the-gpu
float getRandFloat(ivec2 pixelCoords) {
    uvec4 state = imageLoad(uRandomStateImage, pixelCoords);
    float result = stateToFloat(state);
    imageStore(uRandomStateImage, pixelCoords, state);

    return result;
}

vec2 getRandFloat2(ivec2 pixelCoords) {
    uvec4 state = imageLoad(uRandomStateImage, pixelCoords);
    vec2 result;
    result.x = stateToFloat(state);
    result.y = stateToFloat(state);
    imageStore(uRandomStateImage, pixelCoords, state);
    return result;
}

// zAxis should be normalized
mat3 frameZ(vec3 zAxis) {
    mat3 matrix;

    matrix[2] = zAxis;
    if (abs(zAxis.y) > abs(zAxis.x)) {
        float rcpLength = 1.f / length(vec2(zAxis.x, zAxis.y));
        matrix[0] = rcpLength * vec3(zAxis.y, -zAxis.x, 0.f);
    }
    else {
        float rcpLength = 1.f / length(vec2(zAxis.x, zAxis.z));
        matrix[0] = rcpLength * vec3(zAxis.z, 0.f, -zAxis.x);
    }
    matrix[1] = cross(zAxis, matrix[0]);
    return matrix;
}

vec3 uniformSampleHemisphere(float u1, float u2, out float jacobian)
{
    float r = sqrt(1.0 - u1 * u1);
    float phi = 2.0 * M_PI * u2;
    jacobian = 2.0 * M_PI;
    return vec3(cos(phi) * r, sin(phi) * r, u1);
}

vec3 cosineSampleHemisphere(float u1, float u2, out float jacobian)
{
    float r = sqrt(u1);
    float phi = 2.0 * M_PI * u2;
    float cosTheta = sqrt(max(0.0, 1.0 - u1));
    const float x = r * cos(phi);
    const float y = r * sin(phi);
    jacobian = cosTheta > 0.0 ? M_PI / cosTheta : cosTheta;
    return vec3(x, y, cosTheta);
}

vec2 uniformSampleDisk(float radius, float u1, float u2, out float jacobian) {
    float r = sqrt(u1);
    float theta = 2 * M_PI * u2;
    jacobian = M_PI * r * r;
    return radius * r * vec2(cos(theta), sin(theta));
}

float intersectSphere(vec3 org, vec3 dir, Sphere sphere, out vec3 position, out vec3 normal) {
    vec3 centerOrg = org - sphere.center;
    float a = 1;
    float b = 2 * dot(centerOrg, dir);
    float c = dot(centerOrg, centerOrg) - sphere.sqrRadius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0.) {
        return -1.;
    }
    float sqrtDiscriminant = sqrt(discriminant);
    float t1 = 0.5 * (-b - sqrtDiscriminant) / a;
    float t2 = 0.5 * (-b + sqrtDiscriminant) / a;
    float t = t1 >= 0.f ? t1 : t2;

    position = org + t * dir;
    normal = normalize(position - sphere.center);

    return t;
}

float intersectScene(vec3 org, vec3 dir, out vec3 position, out vec3 normal) {
    float currentDist = -1;
    for (uint i = 0; i < uSphereCount; ++i) {
        vec3 tmpPos, tmpNormal;
        float t = intersectSphere(org, dir, sphereArray[i], tmpPos, tmpNormal);
        if (t >= 0.f && (currentDist < 0.f || t < currentDist)) {
            currentDist = t;
            position = tmpPos;
            normal = tmpNormal;
        }
    }
    return currentDist;
}

vec3 ambientOcclusion(vec3 org, vec3 dir, ivec2 pixelCoords) {
    vec3 position, normal;
    vec3 color = vec3(0);
    float dist = intersectScene(org, dir, position, normal);
    if (dist >= 0.) {
        //color = normal;

        mat3 localToWorld = frameZ(normal);
        org = org + dist * dir;
        vec2 uv = getRandFloat2(pixelCoords);
        float jacobian;
        vec3 dir = cosineSampleHemisphere(uv.x, uv.y, jacobian);
        float cosTheta = dir.z;
        dir = localToWorld * dir;
        dist = intersectScene(org + 0.01 * dir, dir, position, normal);
        if (dist < 0.) {
            return vec3(/*jacobian * cosTheta / M_PI*/1); // Works thanks to importance sampling
        }
    }
    return vec3(0.);
}

vec3 normal(vec3 org, vec3 dir, ivec2 pixelCoords) {
    vec3 position, normal;
    vec3 color = vec3(0);
    float dist = intersectScene(org, dir, position, normal);
    if (dist >= 0.) {
        return normal;
    }
    return vec3(0);
}

void main() {
    ivec2 framebufferSize = imageSize(uOutputImage);
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);

    if (pixelCoords.x >= framebufferSize.x || pixelCoords.y >= framebufferSize.y) {
        return;
    }

    vec4 currentEstimate = imageLoad(uAccumImage, pixelCoords);

    vec2 pixelSample = getRandFloat2(pixelCoords);
    float pixelSampleJacobian;
    vec2 diskSample = uniformSampleDisk(1, pixelSample.x, pixelSample.y, pixelSampleJacobian);

    vec2 rasterCoords = vec2(pixelCoords) + vec2(0.5) + diskSample;
    vec2 sampleCoords = rasterCoords / vec2(framebufferSize);

    vec4 ndCoords = vec4(-1, -1, -1, 1) + vec4(2.0 * sampleCoords, 0, 0); // Normalized device coordinates
    vec4 viewCoords = uRcpViewProjMatrix * ndCoords;
    viewCoords /= viewCoords.w;

    vec3 dir = normalize(viewCoords.xyz - uCameraPosition);
    vec3 org = uCameraPosition;

    vec3 color = ambientOcclusion(org, dir, pixelCoords);
    //vec3 color = vec3(getRandFloat(pixelCoords));
    
    vec4 newEstimate = currentEstimate + vec4(color, 1);

    imageStore(uOutputImage, pixelCoords, vec4(newEstimate.xyz / newEstimate.w, 0));
    imageStore(uAccumImage, pixelCoords, newEstimate);
}