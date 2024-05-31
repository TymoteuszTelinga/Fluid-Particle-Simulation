#include "Sandbox.h"
#include <imgui/imgui.h>
#include <iostream>
#include "Renderer/Renderer.h"
#include "Core/Input.h"
#include "Core/SimulationLoader.h"
#include "Core/FileDialog.h"

Sandbox::Sandbox(const ApplicationSpecification& spec, PhysicsSpecification& p_spec, std::vector<obstacle>& obstacles, flow_area in, flow_area out)
	:Application(spec), m_Tint(1.0f), m_Width(spec.Width), m_Height(spec.Height)
{
	m_Camera = CreateScope<Camera>(spec.Width, spec.Height);

	//LoadData("test.yaml");
	
	/*

	int particles_amount = 0;
	int row_amount = int(sqrt(particles_amount));
	m_Particles = CreateRef<Particles>(PARTICLES_LIMIT);

	glm::vec2 s(2, 7);
	int numX = ceil(sqrt(s.x / s.y * particles_amount + (s.x - s.y) * (s.x - s.y) / (4 * s.y * s.y)) - (s.x - s.y) / (2 * s.y));
	int numY = ceil(particles_amount / (float)numX);
	glm::vec2 spawnCentre(3.35, 0.51);

	int i = 0;
	for (int y = 0; y < numY; y++)
	{
		for (int x = 0; x < numX; x++)
		{
			if (i >= particles_amount) {
				break;
			}

			float tx = numX <= 1 ? 0.5f : x / (numX - 1.0f);
			float ty = numY <= 1 ? 0.5f : y / (numY - 1.0f);

			glm::vec2 pos((tx - 0.5f) * s.x, (ty - 0.5f) * s.y);

			pos += spawnCentre;
			m_Particles->addParticle(pos.x, pos.y);
			i++;
		}
	}
	*/
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

	/*
	static int count = 0;
	if (count <= 10) {
		count++;
		return;
	}
	*/

	if (m_Physics && m_Particles)
	{
		for (int i = 0; i < 4; i++) 
		{
			m_Physics->Apply(m_Particles, DeltaTime / 4.0f, 5);
		}
	}

}

void Sandbox::OnRender()
{
	Renderer::BeginScene(*m_Camera);
	if (m_Particles)
	{
		for (int i = 0; i < m_Particles->getSize(); i++) 
		{
			Renderer::DrawQuad(m_Particles->getPosition(i) * m_Physics->getSpecification().MetersToPixel, m_Tint);
		}

		Renderer::DrawQuad(glm::vec2(8, 4) * m_Physics->getSpecification().MetersToPixel, glm::vec3(1, 0, 0));
		Renderer::DrawQuad(glm::vec2(8, 5) * m_Physics->getSpecification().MetersToPixel, glm::vec3(1, 0, 0));
		Renderer::DrawQuad(glm::vec2(9, 5) * m_Physics->getSpecification().MetersToPixel, glm::vec3(1, 0, 0));
		Renderer::DrawQuad(glm::vec2(9, 4) * m_Physics->getSpecification().MetersToPixel, glm::vec3(1, 0, 0));

		Renderer::DrawQuad(glm::vec2(-4, -4) * m_Physics->getSpecification().MetersToPixel, glm::vec3(1, 0, 0));
		Renderer::DrawQuad(glm::vec2(-2,  8) * m_Physics->getSpecification().MetersToPixel, glm::vec3(1, 0, 0));
		Renderer::DrawQuad(glm::vec2(-4,  8) * m_Physics->getSpecification().MetersToPixel, glm::vec3(1, 0, 0));
		Renderer::DrawQuad(glm::vec2(-2, -4) * m_Physics->getSpecification().MetersToPixel, glm::vec3(1, 0, 0));

		Renderer::DrawQuad(glm::vec2(19/2.f, 11/2.f) * m_Physics->getSpecification().MetersToPixel, glm::vec3(0, 1, 0));
		Renderer::DrawQuad(glm::vec2(-19 / 2.f, -11 / 2.f) * m_Physics->getSpecification().MetersToPixel, glm::vec3(0, 1, 0));
		Renderer::DrawQuad(glm::vec2(19 / 2.f, -11 / 2.f) * m_Physics->getSpecification().MetersToPixel, glm::vec3(0, 1, 0));
		Renderer::DrawQuad(glm::vec2(-19 / 2.f, 11 / 2.f) * m_Physics->getSpecification().MetersToPixel, glm::vec3(0, 1, 0));
	}

	Renderer::EndScene();

	/*
	*/

	ImGui::Begin("Test");
	if (ImGui::Button("Open..."))
	{
		std::string path = FileDialog::OpenFile("Layout (*.yaml)\0*.yaml\0");
		if (!path.empty())
		{
			LoadData(path);
		}
		ResetDelta();
	}
	if (ImGui::CollapsingHeader("Stats")) 
	{
		ImGui::Text("Draw calls %d", Renderer::GetStats().DrawCalls);
		ImGui::Text("Quads Count %d", Renderer::GetStats().QuadCount);
		ImGui::Separator();
		ImGui::Text("FPS %d", m_FPS);
		ImGui::Text("Frame Time %f ms", m_FrameTime);
	}
	if (ImGui::CollapsingHeader("Render")) 
	{
		ImGui::DragFloat("Camera speed", &m_CameraSpeed, 1.0f, 5.0f, 500.0f);
		ImGui::Text("Camera zoom: %.2f", m_Camera->GetZoomLevel());
		ImGui::ColorEdit3("Particle", &m_Tint.r);
	}

	if (ImGui::CollapsingHeader("Physics") && m_Physics) 
	{
		PhysicsSpecification& spec = m_Physics->getSpecification();

		ImGui::SeparatorText("Particle");

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

void Sandbox::LoadData(const std::string& filepath)
{
	SimulationLoader Loader;
	Loader.Load(filepath);

	std::vector<obstacle> obstacles = Loader.GetObstacles();
	flow_area in = Loader.GetInArea();
	flow_area out = Loader.GetOutArea();

	PhysicsSpecification spec;
	spec.Width = 19.f;
	spec.Height = 11.f;
	spec.GravityAcceleration = 9.81f;
	spec.CollisionDamping = 0.2;
	spec.KernelRange = 0.35;
	spec.RestDensity = 55;
	spec.GasConstant = 500;
	spec.NearPressureCoef = 18;
	spec.ViscosityStrength = 0.06;

	m_Physics = CreateScope<Physics>(spec, obstacles, in, out);
	m_Particles = CreateRef<Particles>(PARTICLES_LIMIT);
}
