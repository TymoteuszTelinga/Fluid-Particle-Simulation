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

	void Apply(std::vector<Particle>& particles, const float deltaTime) const {
		l_Gravity->Apply(particles, deltaTime);

		for (Particle& particle : particles) {
			particle.Update(deltaTime);
			correct(particle);
		}
	}

private:
	void correct(Particle& particle) const {
		glm::vec2 velocity = particle.GetVelocity();
		glm::vec2 position = particle.GetPosition();

		float min_x = Particle::PARTICLE_RADIUS;
		float max_x = m_specification.Width - Particle::PARTICLE_RADIUS;
		float min_y = Particle::PARTICLE_RADIUS;
		float max_y = m_specification.Height - Particle::PARTICLE_RADIUS;

		if (position.x <= min_x) {
			position.x = min_x + (min_x - position.x);
			velocity.x = -velocity.x*0.8;
		}
		else if (position.x  >= max_x) {
			position.x = max_x + (max_x - position.x);
			velocity.x = -velocity.x*0.8;
		}

		if (position.y <= min_y) {
			position.y = min_y + (min_y - position.y);
			velocity.y = -velocity.y*0.8;
		}
		else if (position.y >= max_y) {
			position.y = max_y + (max_y - position.y);
			velocity.y = -velocity.y*0.8;
		}
		
		particle.SetVelocity(velocity);
		particle.SetPosition(position);
	}

private:
	PhysicsSpecification m_specification;
	Scope<Gravity> l_Gravity;
};

