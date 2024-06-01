#pragma once
#include "Core/Base.h"

#include "Physics/physicsSpecification.h"
#include "Physics/Entities/Particles.h"

/**
* Class for applying gravity force to particles
*
* @note uses CUDA
*/
class Gravity
{
public:

	/**
	* The constructor.
	*
	* @param spec physics specification
	*/
	Gravity(physicsSpecification& spec) : p_spec(spec) {};

	/**
	* applies gravity force
	*
	* @note uses CUDA
	*
	* @param particles for which gravity should be applied.
	* @param deltaTime time from the last physic influence.
	*/
	void Apply(Ref<Particles> particles, float deltaTime) const;

private:
	physicsSpecification& p_spec;
};

