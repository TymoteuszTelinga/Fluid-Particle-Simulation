#include "Physics.h"


void Physics::Apply(std::vector<Particle>& particles, const float deltaTime) const {
	l_Gravity->Apply(particles);
	PredictParticlePositions(particles, 1.0f/120.0f);

	l_NeighbourSearch->UpdateSpatialLookup(particles);
	l_Density->Calculate(particles);
	l_Pressure->Apply(particles);

	Update(particles, deltaTime);
	l_CollisionHandler->Resolve(particles);
}

void Physics::Update(std::vector<Particle>& particles, const float deltaTime) const{
	for (Particle& particle : particles) {
		particle.Update(deltaTime);
	}
}

void Physics::PredictParticlePositions(std::vector<Particle>& particles, const float deltaTime) const {
	for (Particle& p : particles) {
		p.PredictionUpdate(deltaTime);
	}
}