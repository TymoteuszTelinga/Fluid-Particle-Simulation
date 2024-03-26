#pragma once

#include <string>

class Texture
{
private:
    unsigned int m_RendererID;
    std::string FilePath;
    unsigned char* m_LocalBuffer;
    int m_Width, m_Height, m_BPP;//BPP -> bits pre pixel
public:
    Texture(const std::string& Path);
    ~Texture();

    void Bind(unsigned int slot = 0) const;
    void Unbind() const;

    inline int GetRendererID() const { return m_RendererID; };
    inline int GetWidth() const { return m_Width; }
    inline int GetHeight() const { return m_Height; }
};