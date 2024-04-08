#include "Physics.h"


void Physics::Apply(std::vector<Particle>& particles, const float deltaTime) const {
	l_Density->Calculate(particles);
	l_Pressure->Apply(particles, deltaTime);
	//l_Gravity->Apply(particles, deltaTime);

	for (Particle& particle : particles) {
		particle.Update(deltaTime);
		BounceFromBorder(particle);
	}
}

void Physics::BounceFromBorder(Particle& particle) const {
	glm::vec2 velocity = particle.GetVelocity();
	glm::vec2 position = particle.GetPosition();

	float min_x = Particle::RADIUS;
	float max_x = m_specification.Width - Particle::RADIUS;
	float min_y = Particle::RADIUS;
	float max_y = m_specification.Height - Particle::RADIUS;

	if (position.x <= min_x) {
		position.x = min_x + (min_x - position.x);
		velocity.x = -velocity.x * 0.8;
	}
	else if (position.x >= max_x) {
		position.x = max_x + (max_x - position.x);
		velocity.x = -velocity.x * 0.8;
	}

	if (position.y <= min_y) {
		position.y = min_y + (min_y - position.y);
		velocity.y = -velocity.y * 0.8;
	}
	else if (position.y >= max_y) {
		position.y = max_y + (max_y - position.y);
		velocity.y = -velocity.y * 0.8;
	}

	particle.SetVelocity(velocity);
	particle.SetPosition(position);
}