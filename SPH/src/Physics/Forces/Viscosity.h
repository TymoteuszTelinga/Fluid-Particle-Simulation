#pragma once
#include <vector>

#include "Core/Base.h"

#include "Physics/Particle.h"
#include "Physics/PhysicsSpecification.h"
#include "Physics/NeighbourSearch.h"

class Viscosity
{
public:
	Viscosity(PhysicsSpecification& spec, Ref<NeighbourSearch> neighbourSearch) : p_spec(spec), neighbourSearch(neighbourSearch) {}
	void Apply(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours);

private:
	PhysicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
};

