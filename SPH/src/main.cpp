
#include "Sandbox.h"


int main()
{
	ApplicationSpecification spec;
	Sandbox* app = new Sandbox(spec);

	app->Run();

	delete app;

	return 0;
}