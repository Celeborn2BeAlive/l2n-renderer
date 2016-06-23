#version 450
#extension GL_NV_shader_buffer_load : enable
#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 32, local_size_y = 32) in;

#define M_PI 3.14159265358979323846

struct PhongMaterial 
{
    vec4 diffuse;
    vec3 glossy;
    float shininess;
};

struct Sphere 
{
    vec3 center;
    float sqrRadius;
    uint materialID;
    uint padding[3];
};

struct PointLight 
{
    vec3 position;
    float align0;
    vec3 radiantIntensity;
    float align1;
};

struct DirectionalLight 
{
    vec3 incidentDirection;
    float align0;
    vec3 emittedRadiance;
    float align1;
};

uniform uint uIterationCount;
uniform mat4 uRcpViewMatrix;
uniform float uProjRatio;
uniform float uProjTanHalfFovy;
uniform vec3 uCameraPosition;

uniform uint uTileCount;
uniform uint uTileOffset;
uniform ivec2* uTileArray;

uniform uint uMaterialCount;

layout(std430, binding = 0) buffer MaterialBuffer {
    PhongMaterial materialArray[];
};

uniform uint uSphereCount;
uniform Sphere* uSphereArray;

uniform uint uPointLightCount;

layout(std430, binding = 2) buffer PointLightBuffer {
    PointLight pointLightArray[];
};

uniform uint uDirectionalLightCount;

layout(std430, binding = 3) buffer DirectionalLightBuffer {
    DirectionalLight directionalLightArray[];
};

layout(rgba32f) uniform image2D uAccumImage;

writeonly uniform image2D uOutputImage;

/**
* tinymt32 internal state vector and parameters
*/
struct tinymt32_t
{
    uvec4 status;
    uint mat_1;
    uint mat_2;
    uint tmat;
	uint _align0;
};

uniform tinymt32_t* uRandomStateArray;

tinymt32_t tinymt32_load_state(uint pixelIndex)
{
    return uRandomStateArray[pixelIndex];
}

void tinymt32_store_state(uint pixelIndex, tinymt32_t random)
{
    uRandomStateArray[pixelIndex] = random;
}

float tinymt32_generate_floatOO(inout tinymt32_t random);

// zAxis should be normalized
mat3 frameZ(vec3 zAxis) 
{
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
    jacobian = cosTheta > 0.0 ? M_PI / cosTheta : 0.0;
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
    float t1 = 0.5 * (-b - sqrtDiscriminant);
    float t2 = 0.5 * (-b + sqrtDiscriminant);
    float t = (t1 >= 0.f) ? t1 : t2;

    position = org + t * dir;
    normal = normalize(position - sphere.center);

    return t;
}

float intersectSphereVec4(vec3 org, vec3 dir, vec4 sphere, out vec3 position, out vec3 normal) {
    vec3 centerOrg = org - sphere.xyz;
    float a = 1;
    float b = 2 * dot(centerOrg, dir);
    float c = dot(centerOrg, centerOrg) - sphere.w;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0.) {
        return -1.;
    }
    float sqrtDiscriminant = sqrt(discriminant);
    float t1 = 0.5 * (-b - sqrtDiscriminant);
    float t2 = 0.5 * (-b + sqrtDiscriminant);
    float t = (t1 >= 0.f) ? t1 : t2;

    position = org + t * dir;
    normal = normalize(position - sphere.xyz);

    return t;
}

float intersectScene(vec3 org, vec3 dir, out vec3 position, out vec3 normal) {
    float currentDist = -1;
    for (uint i = 0; i < uSphereCount; ++i) {
        vec3 tmpPos, tmpNormal;
        float t = intersectSphere(org, dir, uSphereArray[i], tmpPos, tmpNormal);
        if (t >= 0.f && (currentDist < 0.f || t < currentDist)) {
            currentDist = t;
            position = tmpPos;
            normal = tmpNormal;
        }
    }
    return currentDist;
}

float intersectScene(vec3 org, vec3 dir, out vec3 position, out vec3 normal, out int sphereIndex) {
    float currentDist = -1;
	sphereIndex = -1;
    for (uint i = 0; i < uSphereCount; ++i) {
        vec3 tmpPos, tmpNormal;
        float t = intersectSphere(org, dir, uSphereArray[i], tmpPos, tmpNormal);
        if (t >= 0.f && (currentDist < 0.f || t < currentDist)) {
            currentDist = t;
            position = tmpPos;
            normal = tmpNormal;
            sphereIndex = int(i);
        }
    }
    return currentDist;
}

vec3 getColor(uint n) {
    return fract(
        sin(
            float(n + 1) * vec3(12.9898, 78.233, 56.128)
        )
        * 43758.5453
    );
}

float luminance(const vec3 color) {
    return 0.212671 * color.r + 0.715160 * color.g + 0.072169 * color.b;
}

vec3 pathtracing(vec3 org, vec3 dir, inout tinymt32_t random)
{
    vec3 sunDirection = normalize(vec3(1, 1, -1));

    vec3 position, normal;
    vec3 throughput = vec3(1);
    vec3 color = vec3(0);
    int sphereIndex = -1;
    float dist = intersectScene(org, dir, position, normal, sphereIndex);
    uint pathLength = 0;
    while (dist >= 0.0 && pathLength <= 1) {
        ++pathLength;
        vec3 Kd = getColor(sphereIndex);

        // One sphere on 16 is emissive
        if (sphereIndex % 16 == 0) {
            float sqrRadius = uSphereArray[sphereIndex].sqrRadius;

            float emissionScale = 8192.;
            color += throughput * emissionScale / (4 * M_PI * sqrRadius);
            dist = -2; // Emissive spheres are not reflective
        } else {
            mat3 localToWorld = frameZ(normal);
            org = org + dist * dir;
            vec2 uv = vec2(tinymt32_generate_floatOO(random), tinymt32_generate_floatOO(random));
            float jacobian;
            vec3 localDir = cosineSampleHemisphere(uv.x, uv.y, jacobian);
            float cosTheta = localDir.z;
            dir = localToWorld * localDir;

            throughput *= Kd; // Works thanks to importance sampling, for diffuse spheres

            float rr = tinymt32_generate_floatOO(random);
            float rrProb = min(0.9, luminance(throughput));
            if (rr < rrProb) {
                dist = intersectScene(org + 0.01 * dir, dir, position, normal, sphereIndex);
                throughput /= rrProb;
            } else {
                dist = -2.0;
            }
        }
    }
    // Environment lighting
    if (dist == -1.0 && sphereIndex % 16 != 0)
        color += throughput * 3.f * pow(max(0, dot(sunDirection, dir)), 128);

    return color;
}

vec3 ambientOcclusion(vec3 org, vec3 dir, inout tinymt32_t random)
{
    vec3 position, normal;
    vec3 color = vec3(0);
    float dist = intersectScene(org, dir, position, normal);
    if (dist >= 0.) {
        //color = normal;

        mat3 localToWorld = frameZ(normal);
        org = org + dist * dir;
        vec2 uv = vec2(tinymt32_generate_floatOO(random), tinymt32_generate_floatOO(random));
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

vec3 normal(vec3 org, vec3 dir)
{
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

    uint tileIndex = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    ivec2 tile = uTileArray[(tileIndex + uTileOffset) % uTileCount];

    ivec2 pixelCoords = tile * ivec2(gl_WorkGroupSize.xy) + ivec2(gl_LocalInvocationID.xy);

    if (pixelCoords.x >= framebufferSize.x || pixelCoords.y >= framebufferSize.y) {
        return;
    }

    uint pixelIndex = pixelCoords.x + pixelCoords.y * framebufferSize.x;

    tinymt32_t random = tinymt32_load_state(pixelIndex);

    vec4 currentEstimate = imageLoad(uAccumImage, pixelCoords);

    vec2 pixelSample = vec2(tinymt32_generate_floatOO(random), tinymt32_generate_floatOO(random));
    //float pixelSampleJacobian;
    //vec2 diskSample = uniformSampleDisk(1, pixelSample.x, pixelSample.y, pixelSampleJacobian);

    vec2 rasterCoords = vec2(pixelCoords) + pixelSample;
    vec2 sampleCoords = rasterCoords / vec2(framebufferSize);

    vec4 ndCoords = vec4(-1, -1, 1, 1) + vec4(2.0 * sampleCoords, 0, 0); // Normalized device coordinates
	
	ndCoords *= vec4(uProjRatio * uProjTanHalfFovy, uProjTanHalfFovy, -1, 1); // Equivalent to multiplication by the inverse perspective matrix, but better numerical precision
    vec4 worldCoords = uRcpViewMatrix * ndCoords;

    vec3 dir = normalize(worldCoords.xyz - uCameraPosition);
    vec3 org = uCameraPosition;

    //vec3 color = normal(org, dir);
	//vec3 color = ambientOcclusion(org, dir, random);
    vec3 color = pathtracing(org, dir, random);
    //vec3 color = vec3(getRandFloat(pixelCoords));
    
    vec4 newEstimate = currentEstimate + vec4(color, 1);
    vec3 finalColor = pow(newEstimate.xyz / newEstimate.w, vec3(0.45));

    imageStore(uOutputImage, pixelCoords, vec4(finalColor, 1));
    imageStore(uAccumImage, pixelCoords, newEstimate);

    tinymt32_store_state(pixelIndex, random);
}