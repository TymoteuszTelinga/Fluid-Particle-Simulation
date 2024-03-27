#include "Sandbox.h"
#include <imgui/imgui.h>
#include <iostream>
#include "Renderer/Renderer.h"
#include "Core/Input.h"

Sandbox::Sandbox(const ApplicationSpecification& spec)
	:Application(spec), m_Tint(1.0f)
{
	m_Camera = CreateScope<Camera>(0, spec.Width, 0, spec.Height);

	for (size_t i = 0; i < m_Positions.size(); i++)
	{
		m_Positions[i] = glm::vec2(rand() % spec.Width, rand() % spec.Height);
	}
}

void Sandbox::OnEvent(Event& e)
{
	EventDispatcher Dispacher(e);
	Dispacher.Dispatch<WindowResizeEvent>(BIND_EVENT(Sandbox::Resize));
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
}

void Sandbox::OnRender()
{
	Renderer::BeginScene(*m_Camera);

	for (size_t i = 0; i < m_Positions.size(); i++)
	{
		Renderer::DrawQuad(m_Positions[i],m_Tint);
	}

	Renderer::EndScene();

	ImGui::Begin("Test");
	ImGui::Text("Draw calls %d", Renderer::GetStats().DrawCalls);
	ImGui::Text("Quads Count %d", Renderer::GetStats().QuadCount);
	ImGui::Separator();
	ImGui::Text("FPS %d", m_FPS);
	ImGui::Text("Frame Time %f ms", m_FrameTime);
	ImGui::Separator();
	ImGui::ColorEdit3("Particle", &m_Tint.r);
	ImGui::End();

	Renderer::ResetStats();
	
}

bool Sandbox::Resize(WindowResizeEvent& e)
{
	m_Camera->SetProjection(0, e.GetWidth(), 0, e.GetHeight());
	m_Width = e.GetWidth();
	m_Height = e.GetHeight();

	for (size_t i = 0; i < m_Positions.size(); i++)
	{
		m_Positions[i] = glm::vec2(rand() % m_Width, rand() % m_Height);
	}
	return true;
}
