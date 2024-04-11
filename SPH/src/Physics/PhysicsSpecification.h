#pragma once

#include <stdint.h>

#include "Physics/Kernel.h"

struct PhysicsSpecification {
	uint32_t Width = 800;
	uint32_t Height = 800;
	float MetersToPixel = 100.0f;

	float ParticleRadius = 2.5f;
	float ParticleMass = 1.0f;
	float CollisionDamping = 0.2f;

	float GravityAcceleration = 9.81f;
	float GasConstant = 126.0f;
	float RestDensity = 0.01f;

	KernelFunc DensityKernel = Kernel::Poly6;
	KernelFunc PressureKernelDeriv = Kernel::Spiky3Deriv;
	float KernelRange = 30.0f;
};