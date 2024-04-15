#include "Sandbox.h"
#include <imgui/imgui.h>
#include <iostream>
#include "Renderer/Renderer.h"
#include "Core/Input.h"

Sandbox::Sandbox(const ApplicationSpecification& spec)
	:Application(spec), m_Tint(1.0f), m_Width(spec.Width), m_Height(spec.Height)
{
	m_Camera = CreateScope<Camera>(spec.Width, spec.Height);

	const int halfWidth = spec.Width / 2;
	const int halfHeight = spec.Height / 2;
	for (size_t i = 0; i < m_Positions.size(); i++)
	{
		int x = rand() % spec.Width;
		x -= halfWidth;
		int y = rand() % spec.Height;
		y -= halfHeight;
		m_Positions[i] = glm::vec2(x , y);
	}
}

void Sandbox::OnEvent(Event& e)
{
	EventDispatcher Dispacher(e);
	Dispacher.Dispatch<WindowResizeEvent>(BIND_EVENT(Sandbox::Resize));
	Dispacher.Dispatch<ScrollEvent>(BIND_EVENT(Sandbox::Scroll));
}

void Sandbox::OnUpdate(float DeltaTime)
{

	m_Count++;
	m_CountTime += DeltaTime;

	if (m_CountTime >= 1.0/10.0)
	{
		m_FPS = (1.0 / m_CountTime) * m_Count;
		m_FrameTime = m_CountTime / m_Count * 1000;

		m_CountTime = 0.0f;
		m_Count = 0;
	}

	glm::vec3 cameraPos = m_Camera->GetPosition();
	bool bMoved = false;
	//Movement
	if (Input::IsKeyDown(GLFW_KEY_W))
	{
		cameraPos.y += m_CameraSpeed * DeltaTime;
		bMoved = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_S))
	{
		cameraPos.y -= m_CameraSpeed * DeltaTime;
		bMoved = true;
	}

	if (Input::IsKeyDown(GLFW_KEY_D))
	{
		cameraPos.x += m_CameraSpeed * DeltaTime;
		bMoved = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_A))
	{
		cameraPos.x -= m_CameraSpeed * DeltaTime;
		bMoved = true;
	}

	if (bMoved)
	{
		m_Camera->SetPosition(cameraPos);
	}
}

void Sandbox::OnRender()
{
	Renderer::BeginScene(*m_Camera);

	for (size_t i = 0; i < m_Positions.size(); i++)
	{
		Renderer::DrawQuad(m_Positions[i],m_Tint);
	}
	Renderer::DrawQuad(glm::vec2(0.f,0.f), glm::vec3(1,0,0));

	Renderer::EndScene();

	ImGui::Begin("Test");
	ImGui::Text("Draw calls %d", Renderer::GetStats().DrawCalls);
	ImGui::Text("Quads Count %d", Renderer::GetStats().QuadCount);
	ImGui::Separator();
	ImGui::Text("FPS %d", m_FPS);
	ImGui::Text("Frame Time %f ms", m_FrameTime);
	ImGui::Separator();
	ImGui::DragFloat("Camera speed", &m_CameraSpeed, 1.0f, 5.0f, 500.0f);
	ImGui::Text("Camera zoom: %.2f", m_Camera->GetZoomLevel());
	ImGui::ColorEdit3("Particle", &m_Tint.r);
	ImGui::End();

	Renderer::ResetStats();
	
}

bool Sandbox::Resize(WindowResizeEvent& e)
{
	m_Camera->Resize(e.GetWidth(), e.GetHeight());
	m_Width = e.GetWidth();
	m_Height = e.GetHeight();
	return true;
}

bool Sandbox::Scroll(ScrollEvent& e)
{
	m_Camera->Zoom(-e.GetY()*0.1);
	return true;
}
