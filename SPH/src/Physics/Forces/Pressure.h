#pragma once
#include "Core/Base.h"

#include "Physics/physicsSpecification.h"
#include "Physics/Entities/Particles.h"
#include "Physics/Logic/KernelFactors.h"


/**
* Class for applying pressure force to particles
*
* @note uses CUDA
*/
class Pressure
{
public:

	/**
	* The constructor.
	*
	* @param spec physics specification
	* @param kernel reference to shared class with smoothing kernels factors
	*/
	Pressure(physicsSpecification& spec, Ref<KernelFactors> kernel) : p_spec(spec), kernel(kernel){}

	/**
	* applies pressure force
	*
	* @note uses CUDA
	*
	* @param particles for which pressure should be applied.
	* @param deltaTime time from the last physic influence.
	*/
	void Apply(Ref<Particles> particles, float deltaTime)const;

private:
	physicsSpecification& p_spec;
	Ref<KernelFactors> kernel;
};

