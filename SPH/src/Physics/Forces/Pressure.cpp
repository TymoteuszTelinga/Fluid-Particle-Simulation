#include "Pressure.h"


void Pressure::Apply(std::vector<Particle>& particles) {
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
			glm::vec2 direction = CalculateDirection(center, other, distance);

			float pressureSum = (center.GetPressure() + other.GetPressure());
			float densityProduct = (center.GetDensity() * other.GetDensity());
			glm::vec2 pressureCoef = -p_spec.ParticleMass * pressureSum * 0.5f * direction *  slope;
			center.AddForce(pressureCoef / densityProduct);
			other.AddForce(-pressureCoef / densityProduct);
		}
	}
}

void Pressure::Apply(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours) {
	Particle& center = particles[particleIndex];
	for (size_t j : *neighbours) {
		if (particleIndex == j) {
			continue;
		}
		Particle& other = particles[j];

		float distance = center.calculateDistance(other);
		float slope = p_spec.PressureKernelDeriv(distance, p_spec.KernelRange);
		if (slope == 0) {
			continue;
		}
		glm::vec2 direction = CalculateDirection(center, other, distance);

		float pressureSum = CalculatePressure(center) + CalculatePressure(other);
		float densityProduct = (center.GetDensity() * other.GetDensity());
		glm::vec2 pressureCoef = -p_spec.ParticleMass * pressureSum * 0.5f * direction * slope;
		center.AddForce(pressureCoef / densityProduct);
	}
}

glm::vec2 Pressure::CalculateDirection(const Particle& first, const Particle& second, const float distance)const {
	if (distance == 0) {
		return glm::vec2(1, 0);
	}

	glm::vec2 offset = (second.GetPosition() - first.GetPosition());
	return offset / distance;
}

void Pressure::CalculatePressures(std::vector<Particle>& particles) {
	for (Particle& particle : particles) {
		particle.SetPressure(CalculatePressure(particle));
	}
}

float Pressure::CalculatePressure(const Particle& particle) {
	return p_spec.GasConstant * (particle.GetDensity() - p_spec.RestDensity);
}
