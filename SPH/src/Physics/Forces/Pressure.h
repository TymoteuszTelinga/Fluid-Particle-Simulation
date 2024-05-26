#pragma once
#include <vector>
#include <thread>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particles.h"
#include "Physics/NeighbourSearch.h"

class Pressure
{
public:
	Pressure(PhysicsSpecification& spec, Ref<NeighbourSearch> neighbourSearch, Ref<Kernel> kernel) : p_spec(spec), neighbourSearch(neighbourSearch), kernel(kernel){}
	void Apply(Ref<Particles> particles, float deltaTime)const;

private:
	PhysicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
	Ref<Kernel> kernel;
};

