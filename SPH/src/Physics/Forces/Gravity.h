#pragma once
#include <vector>
#include <thread>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class Gravity
{
public:
	Gravity(PhysicsSpecification& spec) : p_spec(spec) {};
	void Apply(std::vector<Particle>& particles) const;

private:
	Ref<std::thread> RunSubApply(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const;
	void SubApply(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const;

private:
	PhysicsSpecification& p_spec;
};

