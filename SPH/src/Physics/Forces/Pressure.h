#pragma once
#include <vector>

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class Pressure
{
public:
	Pressure(PhysicsSpecification& spec) : p_spec(spec) {}
	void Apply(std::vector<Particle>& particles, float deltaTime);

private:
	void CalculatePressures(std::vector<Particle>& particles);
	float CalculatePressure(const Particle& particle);

	PhysicsSpecification& p_spec;
};

