#pragma once

#include <math.h>
#define PI 3.141592

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
	static constexpr float Poly6Factor = 315/(64*PI);
	static constexpr float SpikyFactor = 15/PI;
	static constexpr float ViscosityFactor = 15 / (2*PI);
	static constexpr float Poly6DerivFactor = 945 / (32 * PI);
	static constexpr float SpikyDerivFactor = 45 / PI;
	static constexpr float ViscosityDerivFactor = 15 / (4 * PI);
};

