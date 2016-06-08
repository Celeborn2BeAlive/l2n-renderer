#version 450
#extension GL_NV_shader_buffer_load : enable

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
    uint padding[3];
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

uvec4 loadTausLCGRandState(ivec2 pixelCoords)
{
    return imageLoad(uRandomStateImage, pixelCoords);
}

void storeTausLCGRandState(ivec2 pixelCoords, uvec4 state)
{
    imageStore(uRandomStateImage, pixelCoords, state);
}

// Hybrid Generator described in  http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
// More details here: http://math.stackexchange.com/questions/337782/pseudo-random-number-generation-on-the-gpu
float TausLCGRand(inout uvec4 state)
{
    state.x = TausStep(state.x, 13, 19, 12, 4294967294);
    state.y = TausStep(state.y, 2, 25, 4, 4294967288);
    state.z = TausStep(state.z, 3, 11, 17, 4294967280);
    state.w = LCGStep(state.w, 1664525, 1013904223);

    return 2.3283064365387e-10 * float(state.x ^ state.y ^ state.z ^ state.w);
}

vec2 TausLCGRand2(inout uvec4 state)
{
    return vec2(TausLCGRand(state), TausLCGRand(state));
}

// Source for tinyMT: https://github.com/MersenneTwister-Lab/TinyMT
layout(rgba32ui) uniform uimage2D uTinyMTRandomStateImage;
layout(rgba32ui) uniform uimage2D uTinyMTRandomMatImage;

#define TINYMT32_MEXP 127
#define TINYMT32_SH0 1
#define TINYMT32_SH1 10
#define TINYMT32_SH8 8
#define TINYMT32_MASK uint(0x7fffffff)
#define TINYMT32_MUL (1.0f / 16777216.0f)

/**
* tinymt32 internal state vector and parameters
*/
struct tinymt32_t
{
    uvec4 status;
    uint mat_1;
    uint mat_2;
    uint tmat;
};

tinymt32_t tinymt32_load_state(ivec2 pixelCoords)
{
    tinymt32_t random;
    random.status = imageLoad(uTinyMTRandomStateImage, pixelCoords);
    uvec4 mat = imageLoad(uTinyMTRandomMatImage, pixelCoords);
    random.mat_1 = mat.x;
    random.mat_2 = mat.y;
    random.tmat = mat.z;
    return random;
}

void tinymt32_store_state(ivec2 pixelCoords, tinymt32_t random)
{
    imageStore(uTinyMTRandomStateImage, pixelCoords, random.status);
    uvec4 tmp = uvec4(random.mat_1, random.mat_2, random.tmat, 0);
    imageStore(uTinyMTRandomMatImage, pixelCoords, tmp);
}

/**
* This function changes internal state of tinymt32.
* Users should not call this function directly.
* @param random tinymt internal status
*/
void tinymt32_next_state(inout tinymt32_t random) {
    uint y = random.status[3];
    uint x = (random.status[0] & TINYMT32_MASK)
        ^ random.status[1]
        ^ random.status[2];
    x ^= (x << TINYMT32_SH0);
    y ^= (y >> TINYMT32_SH0) ^ x;
    random.status[0] = random.status[1];
    random.status[1] = random.status[2];
    random.status[2] = x ^ (y << TINYMT32_SH1);
    random.status[3] = y;
    random.status[1] ^= -(int(y & 1)) & random.mat_1;
    random.status[2] ^= -(int(y & 1)) & random.mat_2;
}

/**
* This function outputs 32-bit unsigned integer from internal state.
* Users should not call this function directly.
* @param random tinymt internal status
* @return 32-bit unsigned pseudorandom number
*/
uint tinymt32_temper(tinymt32_t random) 
{
    uint t0, t1;
    t0 = random.status[3];
#if defined(LINEARITY_CHECK)
    t1 = random->status[0]
        ^ (random->status[2] >> TINYMT32_SH8);
#else
    t1 = random.status[0]
        + (random.status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    t0 ^= -(int(t1 & 1)) & random.tmat;
    return t0;
}

/**
* This function outputs floating point number from internal state.
* Users should not call this function directly.
* @param random tinymt internal status
* @return floating point number r (1.0 <= r < 2.0)
*/
float tinymt32_temper_conv(tinymt32_t random) {
    uint t0, t1;
    uint u;

    t0 = random.status[3];
#if defined(LINEARITY_CHECK)
    t1 = random.status[0]
        ^ (random.status[2] >> TINYMT32_SH8);
#else
    t1 = random.status[0]
        + (random.status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    u = ((t0 ^ (-(int(t1 & 1)) & random.tmat)) >> 9)
        | uint(0x3f800000);
    return uintBitsToFloat(u);
}

/**
* This function outputs floating point number from internal state.
* Users should not call this function directly.
* @param random tinymt internal status
* @return floating point number r (1.0 < r < 2.0)
*/
float tinymt32_temper_conv_open(tinymt32_t random) {
    uint t0, t1;
    uint u;

    t0 = random.status[3];
#if defined(LINEARITY_CHECK)
    t1 = random.status[0]
        ^ (random.status[2] >> TINYMT32_SH8);
#else
    t1 = random.status[0]
        + (random.status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    u = ((t0 ^ (-(int(t1 & 1)) & random.tmat)) >> 9)
        | uint(0x3f800001);
    return uintBitsToFloat(u);
}

/**
* This function outputs 32-bit unsigned integer from internal state.
* @param random tinymt internal status
* @return 32-bit unsigned integer r (0 <= r < 2^32)
*/
uint tinymt32_generate_uint32(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return tinymt32_temper(random);
}

/**
* This function outputs floating point number from internal state.
* This function is implemented using multiplying by (1 / 2^24).
* floating point multiplication is faster than using union trick in
* my Intel CPU.
* @param random tinymt internal status
* @return floating point number r (0.0 <= r < 1.0)
*/
float tinymt32_generate_float(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return (tinymt32_temper(random) >> 8) * TINYMT32_MUL;
}

/**
* This function outputs floating point number from internal state.
* This function is implemented using union trick.
* @param random tinymt internal status
* @return floating point number r (1.0 <= r < 2.0)
*/
float tinymt32_generate_float12(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv(random);
}

/**
* This function outputs floating point number from internal state.
* This function is implemented using union trick.
* @param random tinymt internal status
* @return floating point number r (0.0 <= r < 1.0)
*/
float tinymt32_generate_float01(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv(random) - 1.0f;
}

/**
* This function outputs floating point number from internal state.
* This function may return 1.0 and never returns 0.0.
* @param random tinymt internal status
* @return floating point number r (0.0 < r <= 1.0)
*/
float tinymt32_generate_floatOC(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return 1.0f - tinymt32_generate_float(random);
}

/**
* This function outputs floating point number from internal state.
* This function returns neither 0.0 nor 1.0.
* @param random tinymt internal status
* @return floating point number r (0.0 < r < 1.0)
*/
float tinymt32_generate_floatOO(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv_open(random) - 1.0f;
}

/**
* This function outputs double precision floating point number from
* internal state. The returned value has 32-bit precision.
* In other words, this function makes one double precision floating point
* number from one 32-bit unsigned integer.
* @param random tinymt internal status
* @return floating point number r (0.0 < r <= 1.0)
*/
double tinymt32_generate_32double(inout tinymt32_t random) {
    tinymt32_next_state(random);
    return double(tinymt32_temper(random)) * (1.0 / 4294967296.0);
}

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

float intersectScene(vec3 org, vec3 dir, out vec3 position, out vec3 normal, out uint sphereIndex) {
    float currentDist = -1;
    for (uint i = 0; i < uSphereCount; ++i) {
        vec3 tmpPos, tmpNormal;
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
    uint sphereIndex;
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
            dist = -1; // Emissive spheres are not reflective
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
                dist = -1.0;
            }
        }
    }
	// Environment lighting
    if (pathLength == 0 || sphereIndex % 16 != 0)
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

    tinymt32_t random = tinymt32_load_state(pixelCoords);

    vec4 currentEstimate = imageLoad(uAccumImage, pixelCoords);

    vec2 pixelSample = vec2(tinymt32_generate_floatOO(random), tinymt32_generate_floatOO(random));
    //float pixelSampleJacobian;
    //vec2 diskSample = uniformSampleDisk(1, pixelSample.x, pixelSample.y, pixelSampleJacobian);

    vec2 rasterCoords = vec2(pixelCoords) + pixelSample;
    vec2 sampleCoords = rasterCoords / vec2(framebufferSize);

    vec4 ndCoords = vec4(-1, -1, 1, 1) + vec4(2.0 * sampleCoords, 0, 0); // Normalized device coordinates
    vec4 viewCoords = uRcpViewProjMatrix * ndCoords;
    viewCoords /= viewCoords.w;

    vec3 dir = normalize(viewCoords.xyz - uCameraPosition);
    vec3 org = uCameraPosition;

    //vec3 color = normal(org, dir);
    vec3 color = pathtracing(org, dir, random);
    //vec3 color = vec3(getRandFloat(pixelCoords));
    
    vec4 newEstimate = currentEstimate + vec4(color, 1);
    vec3 finalColor = pow(newEstimate.xyz / newEstimate.w, vec3(0.45));

    imageStore(uOutputImage, pixelCoords, vec4(finalColor, 1));
    imageStore(uAccumImage, pixelCoords, newEstimate);

    tinymt32_store_state(pixelCoords, random);
}