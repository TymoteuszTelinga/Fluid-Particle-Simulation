#pragma once
#include <vector>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class Pressure
{
public:
	Pressure(PhysicsSpecification& spec) : p_spec(spec) {}
	void Apply(std::vector<Particle>& particles);
	void Apply(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours);

private:
	void CalculatePressures(std::vector<Particle>& particles);
	float CalculatePressure(const Particle& particle);
	glm::vec2 CalculateDirection(const Particle& first, const Particle& second, const float distance)const;

	PhysicsSpecification& p_spec;
};

