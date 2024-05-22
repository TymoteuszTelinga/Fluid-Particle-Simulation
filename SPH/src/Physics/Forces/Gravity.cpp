#include "Gravity.h"

void Gravity::Apply(Ref<Particles> particles) const {
	size_t particlesAmount = particles->getSize();
	size_t threadsAmount = 8;
	size_t particlesPerThread = particlesAmount / threadsAmount;

	std::vector<Ref<std::thread>> threads;

	for (int i = 0; i < threadsAmount; i++) {
		size_t firstIndex = i * particlesPerThread;
		threads.push_back(RunSubApply(particles, firstIndex, particlesPerThread));
	}
	if (threadsAmount * particlesPerThread < particlesAmount) {
		threads.push_back(RunSubApply(particles, threadsAmount * particlesPerThread,
			particlesAmount - threadsAmount * particlesPerThread));
	}

	for (Ref<std::thread> thread : threads) {
		thread->join();
	}
}

Ref<std::thread> Gravity::RunSubApply(Ref<Particles> particles, size_t firstIndex, size_t amount)const {
	Ref<std::thread> thread = CreateRef<std::thread>(&Gravity::SubApply, this, particles, firstIndex, amount);
	return thread;
}

void Gravity::SubApply(Ref<Particles> particles, size_t firstIndex, size_t amount) const {
	glm::vec2 gravityForce(0, -p_spec.GravityAcceleration);
	for (size_t index = firstIndex; index < firstIndex + amount; index++) {
		particles->addForce(index, gravityForce);
	}
}
