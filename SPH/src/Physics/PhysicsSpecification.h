#pragma once

#include <stdint.h>

#include "Physics/Kernel.h"

struct PhysicsSpecification {
	uint32_t Width = 800;
	uint32_t Height = 800;
	float MetersToPixel = 100.0f;

	float ParticleRadius = 2.5f;
	float ParticleMass = 1.0f;

	float GravityForce = 9.81f;
	float GasConstant = 8.31f;
	float RestDensity = 0.0001f;

	KernelFunc DensityKernel = Kernel::Poly6;
	KernelFunc PressureKernelDeriv = Kernel::Spiky3Deriv;
	float KernelRange = 10.0f;
};