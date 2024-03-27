#version 460 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 UV;
layout(location = 2) in vec4 Color;

uniform mat4 MVPMatrix;

out vec2 TexUV;
out vec4 TintColor;

void main()
{
   gl_Position = MVPMatrix * position;
   TexUV = UV;
   TintColor = Color;
}