#include "Viscosity.h"

void Viscosity::Apply(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours) {

	Particle& center = particles[particleIndex];
	for (size_t j : *neighbours) {
		if (particleIndex == j) {
			continue;
		}
		Particle& other = particles[j];

		float distance = center.calculatePredictedDistance(other);
		float slope = p_spec.ViscosityKernel(distance, p_spec.KernelRange);

		glm::vec2 velocityDiff = other.GetVelocity() - center.GetVelocity();
		//float densityProduct = (center.GetDensity() * other.GetDensity());
		glm::vec2 viscosityForce = p_spec.ViscosityStrength * p_spec.ParticleMass * velocityDiff * slope;// / densityProduct;
		center.AddForce(viscosityForce);
	}
}