#version 460 core

layout(location = 0) out vec4 color;

in vec2 TexUV;
in vec4 TintColor;

uniform sampler2D ColorMap;

void main()
{
    vec4 texColor = texture(ColorMap, TexUV);

    color = texColor * TintColor;
}