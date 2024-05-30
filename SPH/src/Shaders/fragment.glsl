#version 460 core

layout(location = 0) out vec4 color;

in vec2 TexUV;
in vec4 TintColor;

uniform float Width;

void main()
{
    vec2 pos = (TexUV * 2.f) - 1.f;
    float fade = Width / 4800;//3840
    float distance = 1.0 - length(pos);
    /*
    if (distance < 0.0f)
    {
        discard;
    }
    intensity = vec4(1);
    */
    //vec4 intensity = vec4();
    //intensity.w = 1;
    //TintColor.w = 1;
    
    //if (intensity.a <= 0)
      //  discard;

    color = TintColor;
    color.a = smoothstep(0.0, fade, distance);
}