#include "Sandbox.h"
#include <imgui/imgui.h>
#include <iostream>
#include "Renderer/Renderer.h"
#include "Core/Input.h"
#include "Core/SimulationLoader.h"
#include "Core/FileDialog.h"

Sandbox::Sandbox(const ApplicationSpecification& spec)
	:Application(spec), m_Width(spec.Width), m_Height(spec.Height)
{
	m_Camera = CreateScope<Camera>(spec.Width, spec.Height);

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

	if (m_Physics && m_Particles)
	{
		//for (int i = 0; i < 4; i++) 
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
	}

	Renderer::EndScene();

	//ImGui interface render
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Layout"))
		{
			if (ImGui::MenuItem("Open..."))
			{
				std::string path = FileDialog::OpenFile("Layout (*.yaml)\0*.yaml\0");
				if (!path.empty())
				{
					LoadData(path);
				}
				ResetDelta();
			}

			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Window"))
		{
			if (ImGui::MenuItem("Setting"))
			{
				bSetingIsOpen = true;
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	if (!bSetingIsOpen)
	{
		return;
	}

	ImGui::Begin("Settings", &bSetingIsOpen);
	if (ImGui::CollapsingHeader("Stats")) 
	{
		ImGui::Text("Draw calls %d", Renderer::GetStats().DrawCalls);
		ImGui::Text("Quads Count %d", Renderer::GetStats().ParticleCount);
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
		uint32_t particelLimit = m_Particles->getCapacity();
		if (ImGui::DragScalar("Limit", ImGuiDataType_U32, &particelLimit, 50, 0, &PARTICLES_LIMIT))
		{
			m_Particles->setCapacity(particelLimit);
		}

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
	if (!Loader.Load(filepath))
	{
		return;
	}

	std::vector<obstacle> obstacles = Loader.GetObstacles();
	flow_area in = Loader.GetInArea();
	flow_area out = Loader.GetOutArea();

	PhysicsSpecification spec = Loader.GetSpecification();

	m_Physics = CreateScope<Physics>(spec, obstacles, in, out);
	m_Particles = CreateRef<Particles>(2000,PARTICLES_LIMIT);

	Renderer::ResetRectangles();

	glm::vec2 min = glm::vec2(-(spec.Width / 2.f), -(spec.Height / 2.f)) * spec.MetersToPixel;
	glm::vec2 max = glm::vec2(spec.Width / 2.f, spec.Height / 2.f) * spec.MetersToPixel;
	Renderer::AddRectangle(min, max, glm::vec3(0.1, 0.1, 0.1));

	for (size_t i = 0; i < obstacles.size(); i++)
	{
		glm::vec2 min = glm::vec2(obstacles[i].x_pos, obstacles[i].y_pos) * spec.MetersToPixel;
		glm::vec2 max = min + glm::vec2(obstacles[i].width, obstacles[i].height) * spec.MetersToPixel;
		Renderer::AddRectangle(min, max);
	}

	glm::vec2 outMin = glm::vec2(out.x_pos, out.y_pos) * spec.MetersToPixel;
	glm::vec2 outMax = outMin + glm::vec2(out.width, out.height) * spec.MetersToPixel;
	Renderer::AddRectangle(outMin, outMax, glm::vec3(0.5, 0, 0));

	glm::vec2 inMin = glm::vec2(in.x_pos, in.y_pos) * spec.MetersToPixel;
	glm::vec2 inMax = inMin + glm::vec2(in.width, in.height) * spec.MetersToPixel;
	Renderer::AddRectangle(inMin, inMax, glm::vec3(0, 0.5, 0));

	Renderer::UpdateRectangles();
}
