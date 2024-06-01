#include "Sandbox.h"
#include "Core/SimulationLoader.h"

int main()
{
	ApplicationSpecification spec;

	cudaSetDevice(0);
	Sandbox* app = new Sandbox(spec);
	app->Run();

	delete app;

	return 0;
}