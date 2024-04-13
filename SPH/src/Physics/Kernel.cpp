#include "Kernel.h"

float Kernel::Poly6(float distance, float radius) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return Kernel::Poly6Factor * pow(diff,3) / pow(radius, 8);
	}
	return 0;
}

float Kernel::Spiky3(float distance, float radius) {
	if (distance <= radius) {
		float diff = radius - distance;
		return Kernel::SpikyFactor * pow(diff, 3) / pow(radius, 5);
	}
	return 0;
}


float Kernel::Poly6Deriv(float distance, float radius) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return Kernel::Poly6DerivFactor * distance * diff * diff / pow(radius, 7);
	}
	return 0;
}

float Kernel::Spiky3Deriv(float distance, float radius) {
	if (distance <= radius) {
		float diff = radius - distance;
		return Kernel::SpikyDerivFactor * pow(diff, 2) / pow(radius, 5);
	}
	return 0;
}

float Kernel::ViscosityLaplacian(float distance, float radius) {
	if (distance <= radius) {
		float diff = radius - distance;
		return Kernel::ViscosityLaplacianFactor * diff / pow(radius, 5);
	}
	return 0;
}