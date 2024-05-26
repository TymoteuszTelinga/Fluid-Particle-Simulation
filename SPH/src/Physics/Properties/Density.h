#pragma once
#include <vector>
#include <thread>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particles.h"
#include "Physics/NeighbourSearch.h"

class Density
{

public:
	Density(PhysicsSpecification& spec, Ref<NeighbourSearch> neighbourSearch, Ref<Kernel> kernel) : p_spec(spec), neighbourSearch(neighbourSearch), kernel(kernel){}
	void Calculate(Ref<Particles> particles) const;

private:
	PhysicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
	Ref<Kernel> kernel;
};

