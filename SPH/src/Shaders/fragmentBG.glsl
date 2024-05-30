#version 460 core

layout(location = 0) out vec4 color;

in vec2 TexUV;
in vec4 TintColor;


void main()
{
    color = vec4(vec3(1.f),1.f);
}