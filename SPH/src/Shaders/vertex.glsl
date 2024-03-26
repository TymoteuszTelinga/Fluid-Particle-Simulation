#version 460 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 UV;

uniform mat4 MVPMatrix;

out vec2 TexUV;

void main()
{
   gl_Position = MVPMatrix * position;
   TexUV = UV;
}