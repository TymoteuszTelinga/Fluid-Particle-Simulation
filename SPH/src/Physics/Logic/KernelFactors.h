#pragma once

#include <math.h>
#define PI 3.14159265358979f

typedef float (*KernelFunc)(float, float, float);

class KernelFactors
{
public:
	void updateFactors(float kernelRadius);

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

