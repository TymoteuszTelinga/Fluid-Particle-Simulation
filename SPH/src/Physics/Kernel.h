#pragma once

#include <math.h>
#define PI 3.141592f

typedef float (*KernelFunc)(float, float);

class Kernel
{
public:
	static float Poly6(float distance, float radius);
	static float Spiky3(float distance, float radius);
	static float Viscosity(float distance, float radius);

	static float Poly6Deriv(float distance, float radius);
	static float Spiky3Deriv(float distance, float radius);
	static float ViscosityDeriv(float distance, float radius);

private:
	static constexpr float Poly6Factor = 35/(32*PI);
	static constexpr float SpikyFactor = 2/PI;
	static constexpr float ViscosityFactor = 15 / (2*PI);
	static constexpr float Poly6DerivFactor = 105 / (32 * PI);
	static constexpr float SpikyDerivFactor = 12 / PI;
	static constexpr float ViscosityDerivFactor = 15 / (4 * PI);
};

