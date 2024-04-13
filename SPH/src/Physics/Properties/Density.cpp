#include "Density.h"

void Density::Calculate(std::vector<Particle>& particles) {
	for (int i = 0; i < particles.size(); i++) {
		Particle& center = particles[i];
		center.AddPartialDensity(p_spec.ParticleMass * p_spec.DensityKernel(0.0f, p_spec.KernelRange));
		for (int j = i+1; j < particles.size(); j++) {
			Particle& otherParticle = particles[j];
			float distance = center.calculateDistance(otherParticle);
			float density = p_spec.ParticleMass * p_spec.DensityKernel(distance, p_spec.KernelRange);
			center.AddPartialDensity(density);
			otherParticle.AddPartialDensity(density);
		}
	}
}

void Density::Calculate(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours) {
	Particle& center = particles[particleIndex];
	for (size_t j : *neighbours) {
		Particle& otherParticle = particles[j];
		float distance = center.calculateDistance(otherParticle);
		float density = p_spec.ParticleMass * p_spec.DensityKernel(distance, p_spec.KernelRange);
		center.AddPartialDensity(density);
	}
}



