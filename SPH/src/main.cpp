#include "Sandbox.h"

int main()
{
	ApplicationSpecification spec;
	PhysicsSpecification physSpec;
	physSpec.Width = 17.1f;
	physSpec.Height = 9.3f;
	physSpec.GravityAcceleration = 9.81f;
	physSpec.CollisionDamping = 0.2;
	physSpec.KernelRange = 0.35;
	physSpec.RestDensity = 55;
	physSpec.GasConstant = 500;
	physSpec.NearPressureCoef = 18;
	physSpec.ViscosityStrength = 0.06;

	std::vector<obstacle> obs;
	obstacle os;
	os.height = 12.0f;
	os.width = 2.0f;
	os.x_pos = -4.0f;
	os.y_pos = -4.0f;
	obstacle os2;
	os2.height = 1.0f;
	os2.width = 12.0f;
	os2.x_pos = 0.0f;
	os2.y_pos = -4.0f;

	obs.push_back(os);
	obs.push_back(os2);

	flow_area in;
	in.x_pos = 8.0f;
	in.y_pos = 4.0f;
	in.width = 1.0f;
	in.heigth = 1.0f;
	flow_area out;
	out.x_pos = 8.0f;
	out.y_pos = -5.0f;
	out.width = 1.0f;
	out.heigth = 1.0f;

	cudaSetDevice(0);
	Sandbox* app = new Sandbox(spec, physSpec, obs, in, out);
	app->Run();

	delete app;

	return 0;
}