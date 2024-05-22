#include "Physics.h"


void Physics::Apply(Ref<Particles> particles, const float deltaTime) const {
	l_Gravity->Apply(particles);
	particles->updatePredicted(1.0f / 120.0f);

	l_NeighbourSearch->UpdateSpatialLookup(particles);
	l_Density->Calculate(particles);
	l_Pressure->Apply(particles);
	l_Viscosity->Apply(particles);

	particles->update(deltaTime);
	l_CollisionHandler->Resolve(particles);
}

void Physics::Update(std::vector<Particle>& particles, const float deltaTime) const{
	for (Particle& particle : particles) {
		particle.Update(deltaTime);
	}
}
