#pragma once

#include <math.h>
#define PI 3.14159265358979f

typedef float (*KernelFunc)(float, float);

class Kernel
{
public:
	static float Poly6(float distance, float radius);
	static float Spiky3(float distance, float radius);
	static float Spiky2(float distance, float radius);

	static float Poly6Deriv(float distance, float radius);
	static float Spiky3Deriv(float distance, float radius);
	static float Spiky2Deriv(float distance, float radius);
	static float ViscosityLaplacian(float distance, float radius);

private:
	static constexpr float Poly6Factor = 4/PI;
	static constexpr float Spiky3Factor = 10/PI;
	static constexpr float Spiky2Factor = 6/ PI;
	static constexpr float Poly6DerivFactor = 48/PI;
	static constexpr float Spiky3DerivFactor = 30/PI;
	static constexpr float Spiky2DerivFactor = 12 / PI;
	static constexpr float ViscosityLaplacianFactor = 20/PI;
};

