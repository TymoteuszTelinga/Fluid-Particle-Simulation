#include "Sandbox.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main()
{
	ApplicationSpecification spec;
	PhysicsSpecification physSpec;
	physSpec.Width = 17.1f;
	physSpec.Height = 9.3f;
	physSpec.GravityAcceleration = 12.0f;
	physSpec.CollisionDamping = 0.05;
	physSpec.KernelRange = 0.35;
	physSpec.RestDensity = 55;
	physSpec.GasConstant = 500;
	physSpec.NearPressureCoef = 17.9;
	physSpec.ViscosityStrength = 0.06;

	cudaSetDevice(0);
	Sandbox* app = new Sandbox(spec, physSpec);
	app->Run();

	delete app;

	return 0;
}