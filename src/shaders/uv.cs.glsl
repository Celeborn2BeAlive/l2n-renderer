#version 430

layout(local_size_x = 32, local_size_y = 32) in;

uniform uint uIterationCount;
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
    imageStore(uOutputImage, pixelCoords, vec4(uv * sqrDist, sin(0.02 * float(uIterationCount)), 0));
}