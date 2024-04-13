#include "Kernel.h"

float Kernel::Poly6(float distance, float radius) {
	if (distance <= radius) {
		float diff =  radius * radius - distance * distance;
		return Kernel::Poly6Factor * pow(diff,3) / pow(radius, 7);
	}
	return 0;
}

float Kernel::Spiky3(float distance, float radius) {
	if (distance <= radius) {
		float diff = radius - distance;
		return Kernel::SpikyFactor * pow(diff, 3) / pow(radius, 4);
	}
	return 0;
}


float Kernel::Viscosity(float distance, float radius) {
	if (distance <= radius) {
		float sum = -pow(distance, 3) / (2 * pow(radius, 3))
			+ pow(distance, 2) / pow(radius, 2)
			+ radius / (2 * distance) - 1;
		return Kernel::ViscosityFactor * sum / pow(radius, 3);
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
		return Kernel::SpikyDerivFactor * pow(diff, 2) / pow(radius, 4);
	}
	return 0;
}

float Kernel::ViscosityDeriv(float distance, float radius) {
	if (distance <= radius) {
		float sum = -3 * pow(distance, 2) / pow(radius, 3)
			+ 4 * distance / pow(radius, 2)
			- 1 / pow(distance, 2);
		return  Kernel::ViscosityFactor * sum / pow(radius, 3);
	}
	return 0;
}