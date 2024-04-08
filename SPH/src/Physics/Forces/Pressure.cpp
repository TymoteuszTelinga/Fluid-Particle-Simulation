#include "Pressure.h"

#include <glm/glm.hpp>
#include <stdio.h>

#include "Physics/Properties/Density.h"

void Pressure::Apply(std::vector<Particle>& particles, float deltaTime) {
	CalculatePressures(particles);

	for (int i = 0; i < particles.size()-1; i++) {
		Particle& center = particles[i];
		for (int j = i+1; j < particles.size(); j++) {
			Particle& other = particles[j];

			float distance = center.calculateDistance(other);
			float slope = Pressure::KERNEL_DERIV(distance, Pressure::KERNEL_RADIUS);
			if (slope == 0) {
				continue;
			}

			glm::vec2 direction = (other.GetPosition() - center.GetPosition()) / distance;
			glm::vec2 pressureCoef = -direction * Particle::MASS * slope;
			center.AddForce(pressureCoef * other.GetPressure() / other.GetDensity());
			other.AddForce(-pressureCoef * center.GetPressure() / center.GetDensity());
		}
	}
}

void Pressure::CalculatePressures(std::vector<Particle>& particles) {
	for (Particle& particle : particles) {
		particle.SetPressure(CalculatePressure(particle));
	}
}

float Pressure::CalculatePressure(const Particle& particle) {
	return Pressure::GAS_CONSTANT * (particle.GetDensity() - Density::REST_DENSITY);
}
