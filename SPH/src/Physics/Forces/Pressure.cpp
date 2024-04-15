#include "Pressure.h"

#include <thread>

#include "Core/Base.h"

void Pressure::Apply(std::vector<Particle>& particles)const {
	size_t particlesAmount = particles.size();
	size_t threadsAmount = 8;
	size_t particlesPerThread = particlesAmount / threadsAmount;

	std::vector<Ref<std::thread>> threads;

	for (int i = 0; i < threadsAmount; i++) {
		size_t firstIndex = i * particlesPerThread;
		threads.push_back(RunSubApply(particles, firstIndex, particlesPerThread));
	}
	if (8 * particlesPerThread < particlesAmount) {
		threads.push_back(RunSubApply(particles, threadsAmount * particlesPerThread,
			particlesAmount - threadsAmount * particlesPerThread));
	}

	for (Ref<std::thread> thread : threads) {
		thread->join();
	}
}

Ref<std::thread> Pressure::RunSubApply(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const {
	Ref<std::thread> thread =  CreateRef<std::thread>(&Pressure::SubApply, this, std::ref(particles), firstIndex, amount);
	return thread;
}

void Pressure::SubApply(std::vector<Particle>& particles, size_t firstIndex, size_t amount) const{
	for (size_t index = firstIndex; index < firstIndex + amount; index++) {
		Ref<std::vector<size_t>> neighbours = neighbourSearch->GetParticleNeighbours(particles, index);
		ApplyToParticle(particles, index, neighbours);
	}
}

void Pressure::ApplyToParticle(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours) const{
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

glm::vec2 Pressure::CalculateDirection(const Particle& first, const Particle& second, const float distance) const {
	if (distance == 0) {
		return glm::vec2(1, 0);
	}

	glm::vec2 offset = (second.GetPosition() - first.GetPosition());
	return offset / distance;
}


float Pressure::CalculatePressure(const Particle& particle) const {
	return p_spec.GasConstant * std::max(0.0f,(particle.GetDensity() - p_spec.RestDensity));
}
