#pragma once
#include "Core/Base.h"

#include "Physics/physicsSpecification.h"
#include "Physics/Entities/Particles.h"

class Gravity
{
public:
	Gravity(physicsSpecification& spec) : p_spec(spec) {};
	void Apply(Ref<Particles> particles, float deltaTime) const;

private:
	physicsSpecification& p_spec;
};

