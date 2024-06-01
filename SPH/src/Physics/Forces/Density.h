#pragma once
#include "Core/Base.h"

#include "Physics/physicsSpecification.h"
#include "Physics/Entities/Particles.h"
#include "Physics/Logic/KernelFactors.h"

/**
* Class for calculating particles' density and near density
* 
* @note uses CUDA
*/
class Density
{

public:

	/**
	* The constructor.
	* 
	* @param spec physics specification
	* @param kernel reference to shared class with smoothing kernels factors 
	*/
	Density(physicsSpecification& spec, Ref<KernelFactors> kernel) : p_spec(spec), kernel(kernel){}
	
	/**
	*	calculated density and near density
	* 
	* @note uses CUDA
	* 
	* @param particles for which density and near density will be calculated.
	*/
	void Calculate(Ref<Particles> particles) const;

private:
	physicsSpecification& p_spec;
	Ref<KernelFactors> kernel;
};

