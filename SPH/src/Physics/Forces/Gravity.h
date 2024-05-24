#pragma once
#include <vector>
#include <thread>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particles.h"

class Gravity
{
public:
	Gravity(PhysicsSpecification& spec) : p_spec(spec) {};
	void Apply(Ref<Particles> particles) const;
	void ApplyCuda(Ref<Particles> particles) const;

private:
	Ref<std::thread> RunSubApply(Ref<Particles> particles, size_t firstIndex, size_t amount)const;
	void SubApply(Ref<Particles> particles, size_t firstIndex, size_t amount)const;

private:
	PhysicsSpecification& p_spec;
};

