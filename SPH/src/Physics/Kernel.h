#pragma once

#include <math.h>
#define PI 3.14159265358979f

typedef float (*KernelFunc)(float, float, float);

class Kernel
{
public:
	void updateFactors(float kernelRadius);

	static float Poly6(float distance, float radius, float factor);
	static float Spiky3(float distance, float radius, float factor);
	static float Spiky2(float distance, float radius, float factor);

	static float Poly6Deriv(float distance, float radius, float factor);
	static float Spiky3Deriv(float distance, float radius, float factor);
	static float Spiky2Deriv(float distance, float radius, float factor);
	static float ViscosityLaplacian(float distance, float radius, float factor);

public:
	float Poly6Factor = 0;
	float Spiky3Factor = 0;
	float Spiky2Factor = 0;
	float Poly6DerivFactor = 0;
	float Spiky3DerivFactor = 0;
	float Spiky2DerivFactor = 0;
	float ViscosityLaplacianFactor = 0;

private:
	static constexpr float Poly6ConstFactor = 4 / PI;
	static constexpr float Spiky3ConstFactor = 10 / PI;
	static constexpr float Spiky2ConstFactor = 6 / PI;
	static constexpr float Poly6DerivConstFactor = 48 / PI;
	static constexpr float Spiky3DerivConstFactor = 30 / PI;
	static constexpr float Spiky2DerivConstFactor = 12 / PI;
	static constexpr float ViscosityLaplacianConstFactor = 20 / PI;
};

