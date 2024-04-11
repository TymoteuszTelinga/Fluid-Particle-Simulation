#pragma once
#include <vector>

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class Gravity
{
public:
	Gravity(PhysicsSpecification& spec) : p_spec(spec) {};
	void Apply(std::vector<Particle>& particles, float deltaTime) const;

private:
	PhysicsSpecification& p_spec;
};

