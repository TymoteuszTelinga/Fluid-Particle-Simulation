#pragma once

#include <glm/glm.hpp>

#include<stdio.h>
#include <vector>

#include "Physics/Particle.h"

class Gravity
{

public:
	void Apply(std::vector<Particle>& particles, float deltaTime) const{
		glm::vec2 gravityForce(0,-Gravity::FORCE);
		for (Particle& particle : particles) {
			particle.AddForce(gravityForce);
		}
	}

private:
	static constexpr float FORCE = 200.0f;
};

