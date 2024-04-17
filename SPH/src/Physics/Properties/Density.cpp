#include "Density.h"


void Density::Calculate(std::vector<Particle>& particles)const {
	size_t particlesAmount = particles.size();
	size_t threadsAmount = 8;
	size_t particlesPerThread = particlesAmount / threadsAmount;

	std::vector<Ref<std::thread>> threads;

	for (int i = 0; i < threadsAmount; i++) {
		size_t firstIndex = i * particlesPerThread;
		threads.push_back(RunSubCalculate(particles, firstIndex, particlesPerThread));
	}
	if (8 * particlesPerThread < particlesAmount) {
		threads.push_back(RunSubCalculate(particles, threadsAmount * particlesPerThread, 
			particlesAmount - threadsAmount * particlesPerThread));
	}

	for (Ref<std::thread> thread : threads) {
		thread->join();
	}
}

Ref<std::thread> Density::RunSubCalculate(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const {
	Ref<std::thread> thread = CreateRef<std::thread>(&Density::SubCalculate, this, std::ref(particles), firstIndex, amount);
	return thread;
}

void Density::SubCalculate(std::vector<Particle>& particles, size_t firstIndex, size_t amount) const {
	for (size_t index = firstIndex; index < firstIndex + amount; index++) {
		Ref<std::vector<size_t>> neighbours = neighbourSearch->GetParticleNeighbours(particles, index);
		CalculateForParticle(particles, index, neighbours);
	}
}


void Density::CalculateForParticle(std::vector<Particle>& particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours)const {
	Particle& center = particles[particleIndex];
	for (size_t j : *neighbours) {
		Particle& otherParticle = particles[j];
		float distance = center.calculateDistance(otherParticle);
		float density = p_spec.ParticleMass * p_spec.DensityKernel(distance, p_spec.KernelRange);
		float nearDensity = std::pow((1.0f - distance / p_spec.KernelRange), 3);
		center.AddPartialDensity(density);
		center.AddPartialNearDensity(nearDensity);
	}
}



