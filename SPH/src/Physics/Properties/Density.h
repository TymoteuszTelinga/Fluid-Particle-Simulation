#pragma once
#include <vector>

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class Density
{

public:
	Density(PhysicsSpecification& spec) : p_spec(spec) {}
	void Calculate(std::vector<Particle>& particles);

private:
	PhysicsSpecification& p_spec;
};

