#version 460 core

layout(location = 0) out vec4 color;

in vec2 TexUV;

uniform sampler2D ColorMap;

void main()
{
    vec4 texColor = texture(ColorMap, TexUV);

    color = vec4(texColor);
    //color = vec4(vec3(0,1,0),1.0);
}