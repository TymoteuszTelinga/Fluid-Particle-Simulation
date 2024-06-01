#pragma once
#include "Core/Base.h"

#include "Physics/physicsSpecification.h"
#include "Physics/Entities/Particles.h"
#include "Physics/Logic/KernelFactors.h"


/**
* Class for applying viscosity force to particles
*
* @note uses CUDA
*/
class Viscosity
{
public:

	/**
	* The constructor.
	*
	* @param spec physics specification
	* @param kernel reference to shared class with smoothing kernels factors
	*/
	Viscosity(physicsSpecification& spec, Ref<KernelFactors> kernel) : p_spec(spec), kernel(kernel) {}

	/**
	* applies viscosity force
	*
	* @note uses CUDA
	*
	* @param particles for which viscosity should be applied.
	* @param deltaTime time from the last physic influence.
	*/
	void Apply(Ref<Particles> particles, float deltaTime);

private:
	physicsSpecification& p_spec;
	Ref<KernelFactors> kernel;
};

