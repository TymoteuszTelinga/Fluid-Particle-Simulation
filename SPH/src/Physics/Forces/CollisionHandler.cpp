#include "CollisionHandler.h"

void CollisionHandler::Resolve(std::vector<Particle>& particles) const {
	size_t particlesAmount = particles.size();
	size_t threadsAmount = 8;
	size_t particlesPerThread = particlesAmount / threadsAmount;

	std::vector<Ref<std::thread>> threads;

	for (int i = 0; i < threadsAmount; i++) {
		size_t firstIndex = i * particlesPerThread;
		threads.push_back(RunSubResolve(particles, firstIndex, particlesPerThread));
	}
	if (8 * particlesPerThread < particlesAmount) {
		threads.push_back(RunSubResolve(particles, threadsAmount * particlesPerThread,
			particlesAmount - threadsAmount * particlesPerThread));
	}

	for (Ref<std::thread> thread : threads) {
		thread->join();
	}
}

Ref<std::thread> CollisionHandler::RunSubResolve(std::vector<Particle>& particles, size_t firstIndex, size_t amount) const {
	Ref<std::thread> thread = CreateRef<std::thread>(&CollisionHandler::SubResolve, this, std::ref(particles), firstIndex, amount);
	return thread;
}

void CollisionHandler::SubResolve(std::vector<Particle>& particles, size_t firstIndex, size_t amount)const {
	for (size_t index = firstIndex; index < firstIndex + amount; index++) {
		ResolveCollision(particles[index]);
	}
}


void CollisionHandler::ResolveCollision(Particle& particle) const {
	glm::vec2 velocity = particle.GetVelocity();
	glm::vec2 position = particle.GetPosition();

	float halfWidth = p_spec.Width / 2.0f;
	float halfHeight = p_spec.Height / 2.0f;

	float min_x = -halfWidth + p_spec.ParticleRadius;
	float max_x = halfWidth - p_spec.ParticleRadius;
	float min_y = -halfHeight + p_spec.ParticleRadius;
	float max_y = halfHeight - p_spec.ParticleRadius;

	if (position.x <= min_x) {
		position.x = min_x + (min_x - position.x);
		velocity.x *= -(1 - p_spec.CollisionDamping);
	}
	else if (position.x >= max_x) {
		position.x = max_x + (max_x - position.x);
		velocity.x *= -(1 - p_spec.CollisionDamping);
	}

	if (position.y <= min_y) {
		position.y = min_y + (min_y - position.y);
		velocity.y *= -(1 - p_spec.CollisionDamping);
	}
	else if (position.y >= max_y) {
		position.y = max_y + (max_y - position.y);
		velocity.y *= -(1 - p_spec.CollisionDamping);
	}

	particle.SetVelocity(velocity);
	particle.SetPosition(position);
}