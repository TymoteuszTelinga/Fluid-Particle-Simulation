#include "Kernel.h"
#include "stdio.h"

void Kernel::updateFactors(float kernelRadius) {
	Poly6Factor = Kernel::Poly6ConstFactor / pow(kernelRadius, 8);
	Spiky3Factor = Kernel::Spiky3ConstFactor / pow(kernelRadius, 5);
	Spiky2Factor = Kernel::Spiky2ConstFactor / pow(kernelRadius, 4);
	Poly6DerivFactor = Kernel::Poly6DerivConstFactor / pow(kernelRadius, 8);
	Spiky3DerivFactor = -Kernel::Spiky3DerivConstFactor / pow(kernelRadius, 5);
	Spiky2DerivFactor = -Kernel::Spiky2DerivConstFactor / pow(kernelRadius, 4);
	ViscosityLaplacianFactor = Kernel::ViscosityLaplacianConstFactor / pow(kernelRadius, 5);
}

float Kernel::Poly6(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return factor * pow(diff,3);
	}
	return 0;
}

float Kernel::Spiky3(float distance, float radius, float factor) {
	if (distance < radius) {
		float diff = radius - distance;
		return factor * pow(diff, 3);
	}
	return 0;
}

float Kernel::Spiky2(float distance, float radius, float factor) {
	if (distance < radius) {
		float diff = radius - distance;
		return factor * pow(diff, 2);
	}
	return 0;
}

float Kernel::Poly6Deriv(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return factor * distance * diff * diff;
	}
	return 0;
}

float Kernel::Spiky3Deriv(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff = radius - distance;
		return factor * pow(diff, 2);
	}
	return 0;
}

float Kernel::Spiky2Deriv(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff = radius - distance;
		return factor * diff;
	}
	return 0;
}

float Kernel::ViscosityLaplacian(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff = radius - distance;
		return factor * diff;
	}
	return 0;
}