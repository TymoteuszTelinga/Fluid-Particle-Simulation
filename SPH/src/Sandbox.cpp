#include "Sandbox.h"
#include <imgui/imgui.h>
#include <iostream>
#include "Renderer/Renderer.h"
#include "Core/Input.h"

Sandbox::Sandbox(const ApplicationSpecification& spec, PhysicsSpecification& p_spec)
	:Application(spec), m_Tint(1.0f)
{
	m_Camera = CreateScope<Camera>(0, spec.Width, 0, spec.Height);

	l_Physics = CreateScope<Physics>(p_spec);

	m_Particles.reserve(400);
	for (size_t i = 0; i < m_Particles.capacity(); i++) {
		m_Particles.emplace_back(rand() % spec.Width, rand() % spec.Height);
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
	//printf("%f\n", DeltaTime);
	l_Physics->Apply(m_Particles, 0.01);
}

void Sandbox::OnRender()
{
	Renderer::BeginScene(*m_Camera);
	
	for (Particle& particle : m_Particles) {
		Renderer::DrawQuad(particle.GetPosition(), m_Tint);
	}

	Renderer::EndScene();

	ImGui::Begin("Test");
	if (ImGui::CollapsingHeader("Stats")) {
		ImGui::Text("Draw calls %d", Renderer::GetStats().DrawCalls);
		ImGui::Text("Quads Count %d", Renderer::GetStats().QuadCount);
		ImGui::Separator();
		ImGui::Text("FPS %d", m_FPS);
		ImGui::Text("Frame Time %f ms", m_FrameTime);
	}
	if (ImGui::CollapsingHeader("Render")) {
		ImGui::ColorEdit3("Particle", &m_Tint.r);
		
	}
	ImGui::SetNextItemOpen(true);
	if (ImGui::CollapsingHeader("Physics")) {
		PhysicsSpecification& spec = this->l_Physics->getSpecification();

		ImGui::SeparatorText("Particle");
		ImGui::DragFloat("Particle radius", &spec.ParticleRadius, 0.1f, 0.05f, 1e5);
		ImGui::DragFloat("Particle mass", &spec.ParticleMass, 0.05f, 0.05f, 1e5);

		ImGui::SeparatorText("Forces");
		ImGui::DragFloat("Collision Damping", &spec.CollisionDamping, 0.05f, 0.0f, 1.0f);
		ImGui::DragFloat("Gravity force", &spec.GravityAcceleration, 0.005f, -1e5, 1e5);
		ImGui::DragFloat("Gas constant", &spec.GasConstant, 1.0f, 0.0f, 1e8);
		ImGui::DragFloat("Rest density", &spec.RestDensity, 0.00001f, 0.0f, 1e5, "%.5f");

		ImGui::SeparatorText("Smoothing Kernels");
		ImGui::DragFloat("Kernel range", &spec.KernelRange, 0.005f, 0.05f, 1e5);
	}
	ImGui::End();

	Renderer::ResetStats();
	
}

bool Sandbox::Resize(WindowResizeEvent& e)
{
	m_Camera->SetProjection(0, e.GetWidth(), 0, e.GetHeight());
	m_Width = e.GetWidth();
	m_Height = e.GetHeight();

	//for (Particle& particle : m_Particles) {
	//	particle.SetPosition(glm::vec2(rand() % m_Width, rand() % m_Height));
	//	}

	return true;
}
