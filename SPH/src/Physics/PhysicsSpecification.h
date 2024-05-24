#pragma once
#include <stdint.h>

#include "Physics/Kernel.h"

struct PhysicsSpecification {
	float Width = 8.0f;
	float Height = 8.0f;
	const float MetersToPixel = 100.0f; // 100 px = 1m

	float ParticleRadius = 0.05f; // meters
	float ParticleMass = 1.0f; // kilogram
	float CollisionDamping = 0.2f; // universal unit

	float GravityAcceleration = 9.81f; // m/s^2
	float GasConstant = 5.0f; 
	float RestDensity = 1.5f; // kg/m^3
	float ViscosityStrength = 7.0f; 
	float NearPressureCoef = 5.0f;

	KernelFunc DensityKernel = Kernel::Spiky2;
	KernelFunc NearDensityKernel = Kernel::Spiky3;
	KernelFunc PressureKernelDeriv = Kernel::Spiky2Deriv;
	KernelFunc NearPressureKernelDeriv = Kernel::Spiky3Deriv;
	KernelFunc ViscosityKernel = Kernel::Poly6;
	float KernelRange = 0.5f; // meters
};