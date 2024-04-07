#include "Gravity.h"

void Gravity::Apply(std::vector<Particle>& particles, float deltaTime) const {
	glm::vec2 gravityForce(0, -Gravity::FORCE);
	for (Particle& particle : particles) {
		particle.AddForce(gravityForce);
	}
}