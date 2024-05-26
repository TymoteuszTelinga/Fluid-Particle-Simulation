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
	void Apply(Ref<Particles> particles, float deltaTime) const;

private:
	PhysicsSpecification& p_spec;
};

