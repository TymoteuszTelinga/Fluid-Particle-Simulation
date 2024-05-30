#pragma once
#include <vector>
#include <thread>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"
#include "Physics/NeighbourSearch.h"

class Density
{

public:
	Density(PhysicsSpecification& spec, Ref<NeighbourSearch> neighbourSearch) : p_spec(spec), neighbourSearch(neighbourSearch){}
	void Calculate(std::vector<Particle>& particles) const;

private:
	Ref<std::thread> RunSubCalculate(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const;
	void SubCalculate(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const;
	void CalculateForParticle(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours)const;

private:
	PhysicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
};

