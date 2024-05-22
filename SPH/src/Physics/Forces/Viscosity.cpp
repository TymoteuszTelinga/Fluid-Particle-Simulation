#include "Viscosity.h"

void Viscosity::Apply(Ref<Particles> particles) {
	for (size_t index = 0; index < particles->getSize(); index++) {
		Ref<std::vector<size_t>> neighbours = neighbourSearch->GetParticleNeighbours(particles, index);
		ApplyToParticle(particles, index, neighbours);
	}
}

void Viscosity::ApplyToParticle(Ref<Particles> particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours) {
	for (size_t j : *neighbours) {
		if (particleIndex == j) {
			continue;
		}

		float distance = particles->calculatePredictedDistance(particleIndex, j);
		float slope = p_spec.ViscosityKernel(distance, p_spec.KernelRange, kernel->Poly6Factor);

		glm::vec2 velocityDiff = particles->getVelocity(j) - particles->getVelocity(particleIndex);
		glm::vec2 viscosityForce = p_spec.ViscosityStrength * p_spec.ParticleMass * velocityDiff * slope;
		particles->addForce(particleIndex, viscosityForce);
	}
}