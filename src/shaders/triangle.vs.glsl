#version 330

layout(location = 0) in vec3 iVertexPosition;
layout(location = 1) in vec3 iVertexColor;

out vec3 FragColor;

void main() {
    FragColor = iVertexColor;
    gl_Position = vec4(iVertexPosition, 1.f);
}