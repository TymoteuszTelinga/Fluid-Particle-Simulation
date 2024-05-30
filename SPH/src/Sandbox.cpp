#include "Sandbox.h"
#include <imgui/imgui.h>
#include <iostream>
#include "Renderer/Renderer.h"
#include "Core/Input.h"

Sandbox::Sandbox(const ApplicationSpecification& spec, PhysicsSpecification& p_spec)
	:Application(spec), m_Tint(1.0f), m_Width(spec.Width), m_Height(spec.Height)
{
	m_Camera = CreateScope<Camera>(spec.Width, spec.Height);
	m_Camera->SetPosition(glm::vec3(spec.Width / 2, spec.Height / 2, 0));

	l_Physics = CreateScope<Physics>(p_spec);

	int particles_amount = 500;
	int row_amount = int(sqrt(particles_amount));
	m_Particles.reserve(particles_amount);
	for (size_t i = 0; i < m_Particles.capacity(); i++) {

		//float x_pos = ((float) rand() / (float)RAND_MAX) * p_spec.Width;
		//float y_pos = ((float) rand() / (float)RAND_MAX) * p_spec.Height;
		float x_pos = p_spec.Width/10 + i % row_amount * p_spec.ParticleRadius * 2.0f;
		float y_pos = p_spec.Height/10 + i / row_amount * p_spec.ParticleRadius * 2.0f;
		m_Particles.emplace_back(x_pos, y_pos);
	}

	std::vector<Obstacle> obstacles;
	Obstacle obj;
	obj.Min = glm::vec2(0, -20);
	obj.Max = glm::vec2(spec.Width, 0);
	obstacles.push_back(obj);

	obj.Min = glm::vec2(-20, -20);
	obj.Max = glm::vec2(0, spec.Height);
	obstacles.push_back(obj);

	obj.Min = glm::vec2(spec.Width, -20);
	obj.Max = glm::vec2(spec.Width + 20, spec.Height);
	obstacles.push_back(obj);

	Renderer::SetObstacles(obstacles);
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

	l_Physics->Apply(m_Particles, DeltaTime);
}

void Sandbox::OnRender()
{
	Renderer::BeginScene(*m_Camera);
	
	for (Particle& particle : m_Particles) {
		Renderer::DrawQuad(particle.GetPosition() * l_Physics->getSpecification().MetersToPixel, m_Tint);
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
    ImGui::DragFloat("Camera speed", &m_CameraSpeed, 1.0f, 5.0f, 500.0f);
	  ImGui::Text("Camera zoom: %.2f", m_Camera->GetZoomLevel());
		ImGui::ColorEdit3("Particle", &m_Tint.r);
		
	}

	if (ImGui::CollapsingHeader("Physics")) {
		PhysicsSpecification& spec = this->l_Physics->getSpecification();

		if (ImGui::Button("Reset velocity")) {
			for (Particle& p : m_Particles) {
				p.SetVelocity(glm::vec2(0.0f, 0.0f));
			}
		}
		ImGui::SeparatorText("Particle");
		ImGui::DragFloat("Particle radius", &spec.ParticleRadius, 0.001f, 0.001f, 10.0f);
		ImGui::DragFloat("Particle mass", &spec.ParticleMass, 0.05f, 0.05f, 1e5);

		ImGui::SeparatorText("Forces");
		ImGui::DragFloat("Collision Damping", &spec.CollisionDamping, 0.05f, 0.0f, 1.0f);
		ImGui::DragFloat("Gravity force", &spec.GravityAcceleration, 0.05f, -1e5, 1e5);
		ImGui::DragFloat("Gas constant", &spec.GasConstant, 0.05f, 0.0f, 1e4);
		ImGui::DragFloat("Rest density", &spec.RestDensity, 0.05f, 0.0f, 1e4, "%.2f");
		ImGui::DragFloat("Viscosity Strength", &spec.ViscosityStrength, 0.05f, 0.0f, 1e4, "%.2f");
		ImGui::DragFloat("Near Pressure Coef", &spec.NearPressureCoef, 0.1f, -10.0f, 10.0f, "%.1f");

		ImGui::SeparatorText("Smoothing Kernels");
		ImGui::DragFloat("Kernel range", &spec.KernelRange, 0.01f, 0.05f, 100);
	}
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
