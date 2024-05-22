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
	void Resolve(Ref<Particles> particles) const;

private:
	Ref<std::thread> RunSubResolve(Ref<Particles> particles, size_t firstIndex, size_t amount) const;
	void SubResolve(Ref<Particles> particles, size_t firstIndex, size_t amount)const;
	void ResolveCollision(Ref<Particles> particle, size_t index) const;

private:
private:
	PhysicsSpecification& p_spec;
};

