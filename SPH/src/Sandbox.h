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
#include "Physics/Particles.h"

class Sandbox : public Application
{
public:
	Sandbox(const ApplicationSpecification& spec, PhysicsSpecification& p_spec);
	~Sandbox() {};

	virtual void OnEvent(Event& e) override;
	virtual void OnUpdate(float DeltaTime) override;
	virtual void OnRender() override;

private:
	bool Resize(WindowResizeEvent& e);
	bool Scroll(ScrollEvent& e);

private:
	Scope<Physics> m_Physics;
	//Camera
	Scope<Camera> m_Camera;
	float m_CameraSpeed = 100.0f;


	Ref<Particles> m_Particles;
	glm::vec3 m_Tint;
  
  //Debug info
	float m_FrameTime = 0.0f;
	float m_CountTime = 0.0f;
	int m_FPS = 0;
	int m_Count = 0;

	//
	int m_Width = 640;
	int m_Height = 640;
};