#pragma once
#include <vector>

#include "Core/Base.h"

#include "Physics/Particles.h"
#include "Physics/PhysicsSpecification.h"
#include "Physics/NeighbourSearch.h"

class Viscosity
{
public:
	Viscosity(PhysicsSpecification& spec, Ref<NeighbourSearch> neighbourSearch, Ref<Kernel> kernel) : p_spec(spec), neighbourSearch(neighbourSearch), kernel(kernel) {}
	void Apply(Ref<Particles> particles, float deltaTime);

private:
	PhysicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
	Ref<Kernel> kernel;
};

