#pragma once

#include <GL/glew.h>
#include <string>
#include "Base.h"
#include "Window.h"

struct ApplicationSpecification
{
	std::string Name = "Application";
	uint32_t Width = 1280;
	uint32_t Height = 720;
	uint32_t MinUpdateFrameRate = 30;
};

class Application
{
public:
	Application(const ApplicationSpecification& spec);
	~Application();

	static Application& Get() { return *s_Instance;}

	Window& GetWindow() const { return *m_Window; };

	virtual void Run();

	virtual void OnEvent(Event& e) {};
	virtual void OnUpdate(float DeltaTime) {};
	virtual void OnRender() {};

private:
	void OnEventApp(Event& e);
	bool ResizeVieport(WindowResizeEvent& e);
	//void Init();
	//void Shutdown();

private:
	ApplicationSpecification m_Specification;
	Scope<Window> m_Window;

	bool m_Runing = false;
	float m_DeltaTime = 0.0f;
	float m_LastTime = 0.0f;
	bool m_Minimized = false;

private:
	static Application* s_Instance;
};

