#include "Sandbox.h"

int main()
{
	ApplicationSpecification spec;
	PhysicsSpecification physSpec;
	physSpec.Width = spec.Width / physSpec.MetersToPixel;
	physSpec.Height = spec.Height / physSpec.MetersToPixel;
	physSpec.GravityAcceleration = 0.0f;

	Sandbox* app = new Sandbox(spec, physSpec);
	app->Run();

	delete app;

	return 0;
}