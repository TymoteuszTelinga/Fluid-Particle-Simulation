#include "Sandbox.h"

int main()
{
	ApplicationSpecification spec;
	PhysicsSpecification physSpec;
	physSpec.Width = 20.0f;
	physSpec.Height = 20.0f;
	//physSpec.GravityAcceleration = 0.0f;

	Sandbox* app = new Sandbox(spec, physSpec);
	app->Run();

	delete app;

	return 0;
}