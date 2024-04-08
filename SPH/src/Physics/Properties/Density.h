#pragma once
#include <vector>

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class Density
{

public:
	Density(const PhysicsSpecification& spec) : p_spec(spec) {}
	void Calculate(std::vector<Particle>& particles);

private:
	const PhysicsSpecification& p_spec;
};

