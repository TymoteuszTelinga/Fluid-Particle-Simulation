#pragma once

#include <math.h>
#define PI 3.141592f

typedef float (*KernelFunc)(float, float);

class Kernel
{
public:
	static float Poly6(float distance, float radius);
	static float Spiky3(float distance, float radius);

	static float Poly6Deriv(float distance, float radius);
	static float Spiky3Deriv(float distance, float radius);
	static float ViscosityLaplacian(float distance, float radius);

private:
	static constexpr float Poly6Factor = 8/PI;
	static constexpr float SpikyFactor = 10/PI;
	static constexpr float Poly6DerivFactor = 48/PI;
	static constexpr float SpikyDerivFactor = 30/PI;
	static constexpr float ViscosityLaplacianFactor = 20/PI;
};

