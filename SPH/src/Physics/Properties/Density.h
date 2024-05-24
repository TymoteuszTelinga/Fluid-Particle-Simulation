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
	void CalculateCuda(Ref<Particles> particles) const;

private:
	Ref<std::thread> RunSubCalculate(Ref<Particles> particles, size_t firstIndex, size_t amount)const;
	void SubCalculate(Ref<Particles> particles, size_t firstIndex, size_t amount)const;
	void CalculateForParticle(Ref<Particles> particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours)const;

private:
	PhysicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
	Ref<Kernel> kernel;
};

