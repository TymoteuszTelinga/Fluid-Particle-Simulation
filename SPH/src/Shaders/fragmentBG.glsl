R"(
#version 460 core

layout(location = 0) out vec4 color;

in vec2 TexUV;
in vec4 TintColor;

void main()
{
    color = vec4(TintColor);
}
)"