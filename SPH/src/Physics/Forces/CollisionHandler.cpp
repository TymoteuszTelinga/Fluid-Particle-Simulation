#include "CollisionHandler.h"

void CollisionHandler::Resolve(std::vector<Particle>& particles) const {
	for (Particle& p : particles) {
		ResolveCollision(p);
	}
}

void CollisionHandler::ResolveCollision(Particle& particle) const {
	glm::vec2 velocity = particle.GetVelocity();
	glm::vec2 position = particle.GetPosition();

	float min_x = p_spec.ParticleRadius;
	float max_x = p_spec.Width - p_spec.ParticleRadius;
	float min_y = p_spec.ParticleRadius;
	float max_y = p_spec.Height - p_spec.ParticleRadius;

	if (position.x <= min_x) {
		position.x = min_x + (min_x - position.x);
		velocity.x *= -(1 - p_spec.CollisionDamping);
	}
	else if (position.x >= max_x) {
		position.x = max_x + (max_x - position.x);
		velocity.x *= -(1 - p_spec.CollisionDamping);
	}

	if (position.y <= min_y) {
		position.y = min_y + (min_y - position.y);
		velocity.y *= -(1 - p_spec.CollisionDamping);
	}
	else if (position.y >= max_y) {
		position.y = max_y + (max_y - position.y);
		velocity.y *= -(1 - p_spec.CollisionDamping);
	}

	particle.SetVelocity(velocity);
	particle.SetPosition(position);
}