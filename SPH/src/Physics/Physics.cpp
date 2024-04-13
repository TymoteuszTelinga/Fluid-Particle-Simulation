#include "Physics.h"


void Physics::Apply(std::vector<Particle>& particles, const float deltaTime) const {
	l_Gravity->Apply(particles, deltaTime);
	PredictParticlePositions(particles, 1.0/120);

	l_NeighbourSearch->UpdateSpatialLookup(particles);
	for(int i = 0; i < particles.size(); i++){
		Ref<std::vector<size_t>> neighbours = l_NeighbourSearch->GetParticleNeighbours(particles, i);
		l_Density->Calculate(particles, i, neighbours);
	}

	for (int i = 0; i < particles.size(); i++) {
		Ref<std::vector<size_t>> neighbours = l_NeighbourSearch->GetParticleNeighbours(particles, i);
		l_Pressure->Apply(particles, i, neighbours);
	}

	//for (int i = 0; i < particles.size(); i++) {
	//	Ref<std::vector<size_t>> neighbours = l_NeighbourSearch->GetParticleNeighbours(particles, i);
	//	l_Viscosity->Apply(particles, i, neighbours);
	//}

	//l_Density->Calculate(particles);
	//l_Pressure->Apply(particles);

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