#pragma once
#include <vector>
#include <thread>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class CollisionHandler
{
public:
	CollisionHandler(PhysicsSpecification& spec) : p_spec(spec) {}
	void Resolve(std::vector<Particle>& particles) const;

private:
	Ref<std::thread> RunSubResolve(std::vector<Particle>& particles, size_t firstIndex, size_t amount) const;
	void SubResolve(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const;
	void ResolveCollision(Particle& particle) const;

private:
private:
	PhysicsSpecification& p_spec;
};

