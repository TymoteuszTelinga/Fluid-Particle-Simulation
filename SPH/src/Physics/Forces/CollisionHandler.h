#pragma once
#include <vector>
#include <thread>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particles.h"

class CollisionHandler
{
public:
	CollisionHandler(PhysicsSpecification& spec) : p_spec(spec) {}
	void Resolve(Ref<Particles> particles, float deltaTime) const;

private:
	PhysicsSpecification& p_spec;
};

