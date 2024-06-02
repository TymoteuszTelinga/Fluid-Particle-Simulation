#pragma once
#include <stdint.h>

/**
* Container for physics parameters
*/
struct physicsSpecification {
	const float MetersToPixel = 100.0f; /** constant conversion factor between simulation and rendering units*/
	float Width = 8.0f; /** width of the simulation environment */
	float Height = 8.0f; /** height of the simulation environment */

	float ParticleRadius = 0.05f; /** radius of each particle */
	float CollisionDamping = 0.2f; /** percent of velocity absorbed during collision */

	float GravityAcceleration = 9.81f; /** acceleration of the gravity */
	float GasConstant = 500.0f; /** gas constant, used for calculating pressure */
	float RestDensity = 55.0f; /** density for stationary liquid */
	float ViscosityStrength = 0.06f; /** liquid viscosity factor */ 
	float NearPressureCoef = 18.0f; /** equivalent of gas constant, but for near density */

	float KernelRange = 0.5f; /** range of smoothing kernel */
	float ParticlesPerSecond = 100; /** amount of particles generated in one second*/
};