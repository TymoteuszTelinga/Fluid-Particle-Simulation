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
	void Apply(Ref<Particles> particles)const;
	void ApplyCuda(Ref<Particles> particles)const;

private:
	Ref<std::thread> RunSubApply(Ref<Particles> particles, size_t firstIndex, size_t amount)const;
	void SubApply(Ref<Particles> particles, size_t firstIndex, size_t amount)const;
	void ApplyToParticle(Ref<Particles> particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours)const;

	float CalculatePressure(const float density)const;
	float CalculateNearPressure(const float nearDensity)const;
	glm::vec2 CalculateDirection(const glm::vec2& first, const glm::vec2& second, const float distance)const;

	PhysicsSpecification& p_spec;
	Ref<NeighbourSearch> neighbourSearch;
	Ref<Kernel> kernel;
};

