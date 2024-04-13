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
	glm::vec2 CalculateDirection(const Particle& first, const Particle& second, const float distance)const;

	PhysicsSpecification& p_spec;
};

