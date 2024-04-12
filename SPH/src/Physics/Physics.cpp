#include "Physics.h"


void Physics::Apply(std::vector<Particle>& particles, const float deltaTime) const {
	l_Density->Calculate(particles);
	l_Pressure->Apply(particles, deltaTime);
	l_Gravity->Apply(particles, deltaTime);

	Update(particles, deltaTime);
	l_CollisionHandler->Resolve(particles);
}

void Physics::Update(std::vector<Particle>& particles, const float deltaTime) const{
	for (Particle& particle : particles) {
		particle.Update(deltaTime);
	}
}