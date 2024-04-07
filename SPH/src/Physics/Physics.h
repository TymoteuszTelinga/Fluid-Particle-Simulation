#pragma once

#include <glm/glm.hpp>

#include<stdio.h>
#include <vector>

#include "Core/Base.h"
#include "Physics/Gravity.h"

struct PhysicsSpecification {
	uint32_t Width;
	uint32_t Height;
};

class Physics
{
public:
	Physics(PhysicsSpecification& spec): m_specification(spec) {
		l_Gravity = CreateScope<Gravity>();
	}

	void Apply(std::vector<Particle>& particles, const float deltaTime) const;

private:
	void BounceFromBorder(Particle& particle) const;

private:
	PhysicsSpecification m_specification;
	Scope<Gravity> l_Gravity;
};

