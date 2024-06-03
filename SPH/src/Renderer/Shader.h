#pragma once

#include <unordered_map>
#include <string>
#include <glm/glm.hpp>
#include <glm/detail/type_float.hpp>

class Shader
{
public:
    Shader(const std::string& vertexSrc, const std::string& fragmentSrc);
    ~Shader();

    void Bind() const;
    void Unbind() const;

    //Set uniforms
    void SetUniform1i(const std::string& name, int v0);
    void SetUniform1f(const std::string& name, float v0);
    void SetUniform3f(const std::string& name, float v0, float v1, float v2);
    void SetUniform4f(const std::string& name, float v0, float v1, float v2, float v3);
    void SetUniformMat4f(const std::string& name, const glm::mat4& matrix);
private:
    int GetUniformLocation(const std::string& name);
    std::string loadShaderFromFile(const std::string& filePath) const;
    unsigned int compileShader(unsigned int type, const std::string& source) const;
    unsigned int createShader(const std::string& VertexShader, const std::string& FragmentShader) const;
private:
    unsigned int m_RendererID;
    //cashin for uniforms
    std::unordered_map<std::string, int> m_UniformLocationCache;
};