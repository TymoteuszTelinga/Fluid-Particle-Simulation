#include "Pressure.h"


void Pressure::Apply(std::vector<Particle>& particles, float deltaTime) {
	CalculatePressures(particles);

	for (int i = 0; i < particles.size()-1; i++) {
		Particle& center = particles[i];
		for (int j = i+1; j < particles.size(); j++) {
			Particle& other = particles[j];

			float distance = center.calculateDistance(other);
			float slope = p_spec.PressureKernelDeriv(distance, p_spec.KernelRange);
			if (slope == 0) {
				continue;
			}

			glm::vec2 direction = (other.GetPosition() - center.GetPosition()) / distance;
			glm::vec2 pressureCoef = -direction * p_spec.ParticleMass * slope * (center.GetPressure() + other.GetPressure());
			float density = other.GetDensity() * center.GetDensity();
			center.AddForce(pressureCoef / density);
			other.AddForce(-pressureCoef / density);
		}
	}
}

void Pressure::CalculatePressures(std::vector<Particle>& particles) {
	for (Particle& particle : particles) {
		particle.SetPressure(CalculatePressure(particle));
	}
}

float Pressure::CalculatePressure(const Particle& particle) {
	return p_spec.GasConstant * (particle.GetDensity() - p_spec.RestDensity);
}
