#pragma once

#include <stdint.h>

#include "Physics/Kernel.h"

struct PhysicsSpecification {
	float Width = 8.0f;
	float Height = 8.0f;
	const float MetersToPixel = 100.0f; // 100 px = 1m

	float ParticleRadius = 0.025f; // meters
	float ParticleMass = 1.0f; // kilogram
	float CollisionDamping = 0.2f; // universal unit

	float GravityAcceleration = 9.81f; // m/s^2
	float GasConstant = 20.0f; 
	float RestDensity = 1.0f; // kg/m^3

	KernelFunc DensityKernel = Kernel::Poly6;
	KernelFunc PressureKernelDeriv = Kernel::Spiky3Deriv;
	float KernelRange = 0.5f; // meters
};