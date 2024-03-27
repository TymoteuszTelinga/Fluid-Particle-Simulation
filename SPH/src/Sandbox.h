#pragma once

#include "Core/Application.h"
#include <glm/glm.hpp>

#include <array>

#include "Renderer/Camera.h"
#include "Renderer/VertexArray.h"
#include "Renderer/IndexBuffer.h"
#include "Renderer/Shader.h"
#include "Renderer/Texture.h"

class Sandbox : public Application
{
public:
	Sandbox(const ApplicationSpecification& spec);
	~Sandbox() {};

	virtual void OnEvent(Event& e) override;
	virtual void OnUpdate(float DeltaTime) override;
	virtual void OnRender() override;

private:
	bool Resize(WindowResizeEvent& e);

private:
	Scope<Camera> m_Camera;

	std::array<glm::vec2, 4000> m_Positions;
	glm::vec3 m_Tint;


	float m_Offset = 0.0f;
	float m_FrameTime = 0.0f;
	float m_CountTime = 0.0f;
	int m_FPS = 0;
	int m_Count = 0;

	int m_Width = 640;
	int m_Height = 640;
};