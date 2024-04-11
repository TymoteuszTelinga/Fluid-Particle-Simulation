#pragma once
#include <vector>

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

class CollisionHandler
{
public:
	CollisionHandler(PhysicsSpecification& spec) : p_spec(spec) {}
	void Resolve(std::vector<Particle>& particles) const;

private:
	void ResolveCollision(Particle& particle) const;

private:
private:
	PhysicsSpecification& p_spec;
};

