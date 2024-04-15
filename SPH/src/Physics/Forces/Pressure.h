#pragma once
#include <vector>
#include <thread>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"
#include "Physics/NeighbourSearch.h"

class Pressure
{
public:
	Pressure(PhysicsSpecification& spec, Ref<NeighbourSearch> neighbourSearch) : p_spec(spec), neighbourSearch(neighbourSearch){}
	void Apply(std::vector<Particle>& particles)const;


private:
	Ref<std::thread> RunSubApply(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const;
	void SubApply(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const;
	void ApplyToParticle(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours)const;

	float CalculatePressure(const Particle& particle)const;
	glm::vec2 CalculateDirection(const Particle& first, const Particle& second, const float distance)const;

	PhysicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
};

