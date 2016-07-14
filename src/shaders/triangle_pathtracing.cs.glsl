#version 450
#extension GL_NV_shader_buffer_load : enable
#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 32, local_size_y = 32) in;

#define M_PI 3.14159265358979323846
uint UNDEFINED_INDEX = 0xFFFFFFFF;
float INFINITY = 1.0 / 0.0;
float EPSILON = 0.000001;

uniform uint uIterationCount;
uniform mat4 uRcpViewMatrix;
uniform float uProjRatio;
uniform float uProjTanHalfFovy;
uniform vec3 uCameraPosition;

uniform uint uTileCount;
uniform uint uTileOffset;
uniform ivec2* uTileArray;

struct VertexAttributes
{
    vec3 normal;
    float align0;
    vec2 texCoords;
	float align1[2];
};

uniform uint uMeshCount;
uniform uint* uTriangleCount; // Number of triangle of each mesh
uniform uint* uIndexOffset; // Offset in uIndexBuffer for each mesh

uniform vec4* uVertexBuffer;
uniform VertexAttributes* uVertexAttributesBuffer;
uniform uint* uIndexBuffer; // Each triangle is described by 3 consecutive indices in uVertexBuffer and uVertexAttributesBuffer

layout(rgba32f) uniform image2D uAccumImage;
writeonly uniform image2D uOutputImage;

/*
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

float intersectTriangle(vec3 org, vec3 dir, uint* pVertexIndex, out vec2 uv) 
{
	// Moller-Trumbore algorithm (https://www.wikiwand.com/en/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)
	vec3 v1 = vec3(uVertexBuffer[*(pVertexIndex + 0)]);
    vec3 v2 = vec3(uVertexBuffer[*(pVertexIndex + 1)]);
    vec3 v3 = vec3(uVertexBuffer[*(pVertexIndex + 2)]);

	//Find vectors for two edges sharing V1
	vec3 e1 = v2 - v1;
	vec3 e2 = v3 - v1;

	//Begin calculating determinant - also used to calculate u parameter
	vec3 P = cross(dir, e2);
	float det = dot(e1, P);

	//if determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
	if (abs(det) < EPSILON) {
		return INFINITY;
	}

	float rcpDet = 1. / det;

	//calculate distance from V1 to ray origin
	vec3 T = org - v1;
	float u = dot(T, P) * rcpDet;
	//The intersection lies outside of the triangle
	if(u < 0. || u > 1.) {
		return INFINITY;
	}
	//Prepare to test v parameter
	vec3 Q = cross(T, e1);

	//Calculate V parameter and test bound
	float v = dot(dir, Q) * rcpDet;
	//The intersection lies outside of the triangle
	if(v < 0. || u + v  > 1.) {
		return INFINITY;
	}

	float dist = dot(e2, Q) * rcpDet;
	uv = vec2(u, v);

	return dist < EPSILON ? INFINITY : dist;
}

float intersectScene(vec3 org, vec3 dir, out vec3 position, out vec3 normal, out vec2 texCoords, out uint geomIndex, out uint triangleIndex, out vec2 uv)
{
    float d = INFINITY;

    geomIndex = UNDEFINED_INDEX;
    triangleIndex = UNDEFINED_INDEX;

    uint indexOffset = 0;
    for (uint meshI = 0; meshI < uMeshCount; ++meshI) {
        for (uint triangleI = 0; triangleI < uTriangleCount[meshI]; ++triangleI) {
            vec2 uvIntersect;
            float triangleDist = intersectTriangle(org, dir, uIndexBuffer + uIndexOffset[meshI] + triangleI * 3, uvIntersect);
            if (triangleDist < d) {
                d = triangleDist;
                triangleIndex = triangleI;
                geomIndex = meshI;
                uv = uvIntersect;
            }
        }
    }

    if (d < INFINITY) {
        position = org + dir * d;
        uint indexOffset = uIndexOffset[geomIndex];
        uint i1 = uIndexBuffer[indexOffset + triangleIndex * 3 + 0];
		uint i2 = uIndexBuffer[indexOffset + triangleIndex * 3 + 1];
		uint i3 = uIndexBuffer[indexOffset + triangleIndex * 3 + 2];
        VertexAttributes a = uVertexAttributesBuffer[i1];
        VertexAttributes b = uVertexAttributesBuffer[i2];
        VertexAttributes c = uVertexAttributesBuffer[i3];
        normal = uv.x * b.normal + uv.y * c.normal + (1 - uv.x - uv.y) * a.normal;
        texCoords = uv.x * b.texCoords + uv.y * c.texCoords + (1 - uv.x - uv.y) * a.texCoords;
    }

    return d;
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

vec3 sun_Le(vec3 dir)
{
	vec3 sunDirection = normalize(vec3(1, 1, -1));
	return vec3(pow(max(0, dot(sunDirection, dir)), 128));
}

vec2 complex_sqr(vec2 c)
{
	return vec2(c.x * c.x - c.y * c.y, 2 * c.x * c.y);
}

vec3 mandelbrot_Le(vec3 dir)
{
	float cosTheta = dir.z;
	float sinTheta = sqrt(dot(vec2(dir), vec2(dir)));
	float theta = atan(sinTheta, cosTheta);
	float phi = atan(dir.y, dir.x);
	float u = phi / M_PI;
	float v = -1 + 2 * theta / M_PI;

	vec2 p = 2 * vec2(4, 2) * vec2(u, v);

	vec2 z = vec2(0);
	uint count = 64;
	int i;
	for (i = 0; i < count; ++i) {
		z = complex_sqr(z) + p;

		if (dot(z, z) > 4) {
			break;
		}
	}

	//if (i < 16) {
	//	return vec3(0);
	//}

	if (dot(z, z) > 4) {
		return vec3(i / float(count));
	}

	return vec3(0);
}

vec3 pathtracing(vec3 org, vec3 dir, inout tinymt32_t random)
{
    vec3 position, normal;
	vec2 texCoords;
    vec3 throughput = vec3(1);
    vec3 color = vec3(0);
    uint geomIndex = UNDEFINED_INDEX;
	uint triangleIndex = UNDEFINED_INDEX;
	vec2 uv;
    float dist = intersectScene(org, dir, position, normal, texCoords, geomIndex, triangleIndex, uv);
    uint pathLength = 0;
    while (dist < INFINITY && pathLength <= 1) {
        ++pathLength;
        vec3 Kd = getColor(geomIndex);

        // One sphere on 16 is emissive
        if (geomIndex % 16 == 0) {
            //float sqrRadius = uSphereArray[geomIndex].sqrRadius;
			float sqrRadius = 1.f;

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
            dir = normalize(localToWorld * localDir);

            throughput *= Kd; // Works thanks to importance sampling, for diffuse spheres

            float rr = tinymt32_generate_floatOO(random);
            float rrProb = min(0.9, luminance(throughput));
            if (rr < rrProb) {
                dist = intersectScene(org + 0.01 * dir, dir,  position, normal, texCoords, geomIndex, triangleIndex, uv);
                throughput /= rrProb;
            } else {
                dist = -2.0;
            }
        }
    }
    // Environment lighting
    if (dist == INFINITY && geomIndex % 16 != 0)
        color += throughput * 3.f * mandelbrot_Le(dir);

    return color;
}

vec3 ambientOcclusion(vec3 org, vec3 dir, inout tinymt32_t random)
{
    vec3 position, normal;
	vec2 texCoords;
	uint geomIndex = UNDEFINED_INDEX;
	uint triangleIndex = UNDEFINED_INDEX;
	vec2 uv;
    vec3 color = vec3(0);
    float dist = intersectScene(org, dir, position, normal, texCoords, geomIndex, triangleIndex, uv);
    if (dist < INFINITY) {
        //color = normal;

        mat3 localToWorld = frameZ(normal);
        org = org + dist * dir;
        vec2 uv = vec2(tinymt32_generate_floatOO(random), tinymt32_generate_floatOO(random));
        float jacobian;
        vec3 dir = cosineSampleHemisphere(uv.x, uv.y, jacobian);
        float cosTheta = dir.z;
        dir = localToWorld * dir;
        dist = intersectScene(org + 0.01 * dir, dir, position, normal, texCoords, geomIndex, triangleIndex, uv);
        if (dist < INFINITY) {
            return vec3(/*jacobian * cosTheta / M_PI*/1); // Works thanks to importance sampling
        }
    }
    return vec3(0.);
}

vec3 normal(vec3 org, vec3 dir)
{
    vec3 position, normal;
	vec2 texCoords;
	uint geomIndex = UNDEFINED_INDEX;
	uint triangleIndex = UNDEFINED_INDEX;
	vec2 uv;
    vec3 color = vec3(0);
    float dist = intersectScene(org, dir, position, normal, texCoords, geomIndex, triangleIndex, uv);
    if (dist < INFINITY) {
        return normal;
    }
    return vec3(1, 0, 1);
}

vec3 texCoords(vec3 org, vec3 dir)
{
    vec3 position, normal;
	vec2 texCoords;
	uint geomIndex = UNDEFINED_INDEX;
	uint triangleIndex = UNDEFINED_INDEX;
	vec2 uv;
    vec3 color = vec3(0);
    float dist = intersectScene(org, dir, position, normal, texCoords, geomIndex, triangleIndex, uv);
    if (dist < INFINITY) {
        return vec3(texCoords, 0);
    }
    return vec3(1, 0, 1);
}

vec3 paramUV(vec3 org, vec3 dir)
{
    vec3 position, normal;
	vec2 texCoords;
	uint geomIndex = UNDEFINED_INDEX;
	uint triangleIndex = UNDEFINED_INDEX;
	vec2 uv;
    vec3 color = vec3(0);
    float dist = intersectScene(org, dir, position, normal, texCoords, geomIndex, triangleIndex, uv);
    if (dist < INFINITY) {
        return vec3(uv, 0);
    }
    return vec3(1, 0, 1);
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

    //vec3 color = texCoords(org, dir);
	//vec3 color = ambientOcclusion(org, dir, random);
    vec3 color = pathtracing(org, dir, random);
    //vec3 color = vec3(getRandFloat(pixelCoords));
    
    vec4 newEstimate = currentEstimate + vec4(color, 1);
    vec3 finalColor = pow(newEstimate.xyz / newEstimate.w, vec3(0.45));

    imageStore(uOutputImage, pixelCoords, vec4(finalColor, 1));
    imageStore(uAccumImage, pixelCoords, newEstimate);

    tinymt32_store_state(pixelIndex, random);
}