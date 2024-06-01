#pragma once
#include "Core/Base.h"

#include "Physics/physicsSpecification.h"
#include "Physics/Entities/Particles.h"
#include "Physics/Logic/NeighbourSearch.h"
#include "Physics/Logic/KernelFactors.h"

class Density
{

public:
	Density(physicsSpecification& spec, Ref<NeighbourSearch> neighbourSearch, Ref<KernelFactors> kernel) : p_spec(spec), neighbourSearch(neighbourSearch), kernel(kernel){}
	void Calculate(Ref<Particles> particles) const;

private:
	physicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
	Ref<KernelFactors> kernel;
};

