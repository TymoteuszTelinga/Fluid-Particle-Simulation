#pragma once
#include <stdint.h>

struct physicsSpecification {
	float Width = 8.0f;
	float Height = 8.0f;
	const float MetersToPixel = 100.0f; // 100 px = 1m

	float ParticleRadius = 0.05f; // meters
	float CollisionDamping = 0.2f; // universal unit

	float GravityAcceleration = 9.81f; // m/s^2
	float GasConstant = 500.0f; 
	float RestDensity = 55.0f; // kg/m^3
	float ViscosityStrength = 0.06f; 
	float NearPressureCoef = 18.0f;

	float KernelRange = 0.5f; // meters
};