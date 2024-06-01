#pragma once

#include "Core/Application.h"
#include <glm/glm.hpp>

#include <array>
#include <vector>

#include "Renderer/Camera.h"
#include "Renderer/VertexArray.h"
#include "Renderer/IndexBuffer.h"
#include "Renderer/Shader.h"
#include "Renderer/Texture.h"

#include "Physics/Physics.h"
//#include "Physics/Entities/Particles.h"

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
	bool Scroll(ScrollEvent& e);
	void LoadData(const std::string& filepath);

private:
	Scope<Physics> m_Physics = nullptr;
	//Camera
	Scope<Camera> m_Camera;
	float m_CameraSpeed = 400.0f;


	Ref<Particles> m_Particles = nullptr;
	glm::vec3 m_Tint = glm::vec3(0.260f, 0.739f, 1.000f);

	bool bSetingIsOpen = false;
  
	//Debug info
	float m_FrameTime = 0.0f;
	float m_CountTime = 0.0f;
	int m_FPS = 0;
	int m_Count = 0;

	//
	int m_Width = 640;
	int m_Height = 640;
};