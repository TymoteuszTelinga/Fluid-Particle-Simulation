#pragma once
#include <vector>

#include "Core/Base.h"

#include "Physics/Particle.h"
#include "Physics/PhysicsSpecification.h"

class Viscosity
{
public:
	Viscosity(PhysicsSpecification& spec): p_spec(spec){}
	void Apply(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours);

private:
	PhysicsSpecification& p_spec;
};

