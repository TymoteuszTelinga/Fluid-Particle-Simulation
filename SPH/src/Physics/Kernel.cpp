#include "Kernel.h"

float Kernel::Poly6(float distance, float radius) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return Kernel::Poly6Factor * pow(diff,3) / pow(radius, 8);
	}
	return 0;
}

float Kernel::Spiky3(float distance, float radius) {
	if (distance < radius) {
		float diff = radius - distance;
		return Kernel::Spiky3Factor * pow(diff, 3) / pow(radius, 5);
	}
	return 0;
}

float Kernel::Spiky2(float distance, float radius) {
	if (distance < radius) {
		float diff = radius - distance;
		return Kernel::Spiky2Factor * pow(diff, 2) / pow(radius, 4);
	}
	return 0;
}


float Kernel::Poly6Deriv(float distance, float radius) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return Kernel::Poly6DerivFactor * distance * diff * diff / pow(radius, 8);
	}
	return 0;
}

float Kernel::Spiky3Deriv(float distance, float radius) {
	if (distance <= radius) {
		float diff = radius - distance;
		return -Kernel::Spiky3DerivFactor * pow(diff, 2) / pow(radius, 5);
	}
	return 0;
}

float Kernel::Spiky2Deriv(float distance, float radius) {
	if (distance <= radius) {
		float diff = radius - distance;
		return -Kernel::Spiky2DerivFactor * diff / pow(radius, 4);
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