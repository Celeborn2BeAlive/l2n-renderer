#version 430

layout(local_size_x = 32, local_size_y = 32) in;

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

writeonly uniform image2D uOutputImage;

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

void main() {
    ivec2 framebufferSize = imageSize(uOutputImage);
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);

    if (pixelCoords.x >= framebufferSize.x || pixelCoords.y >= framebufferSize.y) {
        return;
    }

    vec3 finalColor = vec3(0);

    vec2 rasterCoords = (vec2(pixelCoords) + vec2(0.5)) / vec2(framebufferSize); // Center of pixel between 0 and 1 for each axis
    vec4 ndCoords = vec4(-1, -1, -1, 1) + vec4(2.0 * rasterCoords, 0, 0); // Normalized device coordinates
    vec4 viewCoords = uRcpViewProjMatrix * ndCoords;
    viewCoords /= viewCoords.w;

    //vec4 viewCoords = vec4(rasterCoords, -1, 0);

    vec3 dir = normalize(viewCoords.xyz - uCameraPosition);
    vec3 org = uCameraPosition;

    vec3 position, normal;
    float dist = intersectScene(org, dir, position, normal);
    if (dist >= 0.f) {
        finalColor = normal;
    }
    
    imageStore(uOutputImage, pixelCoords, vec4(finalColor, 0));
}