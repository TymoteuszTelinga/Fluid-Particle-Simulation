#include "Density.h"

#include "Cuda/Kernels.cuh"

void Density::CalculateCuda(Ref<Particles> particles) const {
	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	int cellCols = (int)(p_spec.Height / p_spec.KernelRange) + 1;

	DensityCuda(particles->c_predictedPositions_x, particles->c_predictedPositions_y, particles->c_densities,
		particles->c_nearDensities, p_spec.KernelRange, kernel->Spiky2Factor, kernel->Spiky3Factor,
		particles->getSize(), particles->c_lookup_index, particles->c_lookup_key, cellRows, cellCols, particles->c_indices, particles->c_indices_size);
}

void Density::Calculate(Ref<Particles> particles)const {
	size_t particlesAmount = particles->getSize();
	size_t threadsAmount = 8;
	size_t particlesPerThread = particlesAmount / threadsAmount;

	std::vector<Ref<std::thread>> threads;

	for (int i = 0; i < threadsAmount; i++) {
		size_t firstIndex = i * particlesPerThread;
		threads.push_back(RunSubCalculate(particles, firstIndex, particlesPerThread));
	}
	if (threadsAmount * particlesPerThread < particlesAmount) {
		threads.push_back(RunSubCalculate(particles, threadsAmount * particlesPerThread, 
			particlesAmount - threadsAmount * particlesPerThread));
	}

	for (Ref<std::thread> thread : threads) {
		thread->join();
	}
}

Ref<std::thread> Density::RunSubCalculate(Ref<Particles> particles, size_t firstIndex, size_t amount)const {
	Ref<std::thread> thread = CreateRef<std::thread>(&Density::SubCalculate, this, particles, firstIndex, amount);
	return thread;
}

void Density::SubCalculate(Ref<Particles> particles, size_t firstIndex, size_t amount) const {
	for (size_t index = firstIndex; index < firstIndex + amount; index++) {
		Ref<std::vector<size_t>> neighbours = neighbourSearch->GetParticleNeighbours(particles, index);
		CalculateForParticle(particles, index, neighbours);
	}
}


void Density::CalculateForParticle(Ref<Particles> particles, size_t particleIndex, Ref<std::vector<size_t>> neighbours)const {
	for (size_t j : *neighbours) {
		float distance = particles->calculatePredictedDistance(particleIndex, j);
		float density = p_spec.ParticleMass * p_spec.DensityKernel(distance, p_spec.KernelRange, kernel->Spiky2Factor);
		float nearDensity = p_spec.NearDensityKernel(distance, p_spec.KernelRange, kernel->Spiky3Factor);
		particles->addDensity(particleIndex, density);
		particles->addNearDensity(particleIndex, nearDensity);
	}
}



