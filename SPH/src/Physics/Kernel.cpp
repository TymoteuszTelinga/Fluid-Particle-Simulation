#include "Kernel.h"

float Kernel::Poly6(float distance, float radius) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return pow(diff,3) * Kernel::Poly6Factor / pow(radius, 9);
	}
	return 0;
}

float Kernel::Spiky3(float distance, float radius) {
	if (distance <= radius) {
		float diff = radius - distance;
		return pow(diff, 3) * Kernel::SpikyFactor / pow(radius, 6);
	}
	return 0;
}


float Kernel::Viscosity(float distance, float radius) {
	if (distance <= radius) {
		float sum = -pow(distance, 3) / (2 * pow(radius, 3))
			+ pow(distance, 2) / pow(radius, 2)
			+ radius / (2 * distance) - 1;
		return sum * Kernel::ViscosityFactor / pow(radius, 3);
	}
	return 0;
}

float Kernel::Poly6Deriv(float distance, float radius) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return distance * diff * diff * Kernel::Poly6DerivFactor / pow(radius, 9);
	}
	return 0;
}

float Kernel::Spiky3Deriv(float distance, float radius) {
	if (distance <= radius) {
		float diff = radius - distance;
		return pow(diff, 2) * Kernel::SpikyDerivFactor / pow(radius, 6);
	}
	return 0;
}

float Kernel::ViscosityDeriv(float distance, float radius) {
	if (distance <= radius) {
		float sum = -3 * pow(distance, 2) / pow(radius, 3)
			+ 4 * distance / pow(radius, 2)
			- 1 / pow(distance, 2);
		return sum * Kernel::ViscosityFactor / pow(radius, 3);
	}
	return 0;
}