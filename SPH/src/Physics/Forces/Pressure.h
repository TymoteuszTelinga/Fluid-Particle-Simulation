#pragma once
#include <vector>

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class Pressure
{
public:
	Pressure(const PhysicsSpecification& spec) : p_spec(spec) {}
	void Apply(std::vector<Particle>& particles, float deltaTime);

private:
	void CalculatePressures(std::vector<Particle>& particles);
	float CalculatePressure(const Particle& particle);

	const PhysicsSpecification& p_spec;
};

