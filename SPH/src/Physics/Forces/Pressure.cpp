#include "Pressure.h"

#include <thread>

#include "Core/Base.h"

void Pressure::Apply(Ref<Particles> particles)const {
	size_t particlesAmount = particles->getSize();
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

Ref<std::thread> Pressure::RunSubApply(Ref<Particles> particles, size_t firstIndex, size_t amount)const {
	Ref<std::thread> thread =  CreateRef<std::thread>(&Pressure::SubApply, this, particles, firstIndex, amount);
	return thread;
}

void Pressure::SubApply(Ref<Particles> particles, size_t firstIndex, size_t amount) const{
	for (size_t index = firstIndex; index < firstIndex + amount; index++) {
		Ref<std::vector<size_t>> neighbours = neighbourSearch->GetParticleNeighbours(particles, index);
		ApplyToParticle(particles, index, neighbours);
	}
}

void Pressure::ApplyToParticle(Ref<Particles> particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours) const{
	for (size_t j : *neighbours) {
		if (particleIndex == j) {
			continue;
		}

		float distance = particles->calculatePredictedDistance(particleIndex, j);
		float slope = p_spec.PressureKernelDeriv(distance, p_spec.KernelRange);
		float nearSlope = p_spec.NearPressureKernelDeriv(distance, p_spec.KernelRange);
		if (slope == 0) {
			continue;
		}
		glm::vec2 direction = CalculateDirection(particles->getPredictedPosition(particleIndex), particles->getPredictedPosition(j), distance);
		float pressureShare = (CalculatePressure(particles->getDensity(particleIndex)) + CalculatePressure(particles->getDensity(j))) * 0.5f;
		float nearPressureShare = (CalculateNearPressure(particles->getNearDensity(particleIndex)) + CalculateNearPressure(particles->getNearDensity(j))) * 0.5f;
		float densityProduct = particles->getDensity(particleIndex) * particles->getDensity(j);
		glm::vec2 pressureCoef = p_spec.ParticleMass * pressureShare * direction * slope;
		glm::vec2 nearPressureCoef = p_spec.ParticleMass * nearPressureShare * direction * nearSlope;
		particles->addForce(particleIndex, pressureCoef / densityProduct);
		particles->addForce(particleIndex, nearPressureCoef / (particles->getDensity(particleIndex) * particles->getNearDensity(j)));
	}
}

glm::vec2 Pressure::CalculateDirection(const glm::vec2& first, const glm::vec2& second, const float distance) const {
	if (distance <= 0.0f) {
		return glm::vec2(0, 1);
	}

	glm::vec2 offset = (second - first);
	return offset / distance;
}


float Pressure::CalculatePressure(const float density) const {
	return p_spec.GasConstant * (density - p_spec.RestDensity);
}

float Pressure::CalculateNearPressure(const float nearDensity) const {
	return nearDensity * p_spec.NearPressureCoef;
}
