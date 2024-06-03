R"(
#version 460 core

layout(location = 0) out vec4 color;

in vec2 TexUV;
in vec4 TintColor;

uniform float Width;

void main()
{
    vec2 pos = (TexUV * 2.f) - 1.f;
    float fade = Width / 4800;
    float distance = 1.0 - length(pos);

    color = TintColor;
    color.a = smoothstep(0.0, fade, distance);
}
)"