#include "Particles.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Cuda/Kernels.cuh"


Particles::Particles(size_t capacity) : capacity(capacity), size(0)
{
	positions_x = new float[capacity];
	positions_y = new float[capacity];

	predictedPositions_x = new float[capacity];
	predictedPositions_y = new float[capacity];

	velocities_x = new float[capacity];
	velocities_y = new float[capacity];

	forces_x = new float[capacity];
	forces_y = new float[capacity];

	densities = new float[capacity];
	nearDensities = new float[capacity];

	cudaMalloc(&c_positions_x, capacity * sizeof(float));
	cudaMalloc(&c_positions_y, capacity * sizeof(float));
	cudaMalloc(&c_predictedPositions_x, capacity * sizeof(float));
	cudaMalloc(&c_predictedPositions_y, capacity * sizeof(float));
	cudaMalloc(&c_velocities_x, capacity * sizeof(float));
	cudaMalloc(&c_velocities_y, capacity * sizeof(float));
	cudaMalloc(&c_forces_x, capacity * sizeof(float));
	cudaMalloc(&c_forces_y, capacity * sizeof(float));
	cudaMalloc(&c_densities, capacity * sizeof(float));
	cudaMalloc(&c_nearDensities, capacity * sizeof(float));

	cudaMalloc(&c_lookup_index, capacity * sizeof(int));
	cudaMalloc(&c_lookup_key, capacity * sizeof(int));
	cudaMalloc(&c_indices, capacity * sizeof(int));
}

Particles::~Particles() {
	delete[] positions_x;
	delete[] positions_y;

	delete[] predictedPositions_x;
	delete[] predictedPositions_y;

	delete[] velocities_x;
	delete[] velocities_y;

	delete[] forces_x;
	delete[] forces_y;

	delete[] densities;
	delete[] nearDensities;

	cudaFree(c_positions_x);
	cudaFree(c_positions_y);
	cudaFree(c_predictedPositions_x);
	cudaFree(c_predictedPositions_y);
	cudaFree(c_velocities_x);
	cudaFree(c_velocities_y);
	cudaFree(c_forces_x);
	cudaFree(c_forces_y);
	cudaFree(c_densities);
	cudaFree(c_nearDensities);

	cudaFree(c_lookup_index);
	cudaFree(c_lookup_key);
	cudaFree(c_indices);
}
void Particles::sendToCuda() {
	cudaMemcpy(c_positions_x, positions_x, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_positions_y, positions_y, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_predictedPositions_x, predictedPositions_x, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_predictedPositions_y, predictedPositions_y, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_velocities_x, velocities_x, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_velocities_y, velocities_y, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_forces_x, forces_x, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_forces_y, forces_y, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_densities, densities, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_nearDensities, nearDensities, size * sizeof(float), cudaMemcpyHostToDevice);
}

void Particles::getFromCudaBeforeSpatial() {
	cudaMemcpy(predictedPositions_x, c_predictedPositions_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(predictedPositions_y, c_predictedPositions_y, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Particles::getFromCuda() {
	cudaMemcpy(positions_x, c_positions_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(positions_y, c_positions_y, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(predictedPositions_x, c_predictedPositions_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(predictedPositions_y, c_predictedPositions_y, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(velocities_x, c_velocities_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(velocities_y, c_velocities_y, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(forces_x, c_forces_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(forces_y, c_forces_y, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(densities, c_densities, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(nearDensities, c_nearDensities, size * sizeof(int), cudaMemcpyDeviceToHost);
}

void Particles::updatePredictedCuda(float deltaTime) {
	UpdatePredictedFromCuda(c_velocities_x, c_velocities_y, c_forces_x, c_forces_y,
		c_predictedPositions_x, c_predictedPositions_y, c_positions_x, c_positions_y, deltaTime, size);
}

bool Particles::addParticle(float x_pos, float y_pos) {
	if (size >= capacity) {
		return false;
	}

	positions_x[size] = x_pos;
	positions_y[size] = y_pos;
	predictedPositions_x[size] = x_pos;
	predictedPositions_y[size] = y_pos;
	velocities_x[size] = 0;
	velocities_y[size] = 0;
	forces_x[size] = 0;
	forces_y[size] = 0;
	densities[size] = 0;
	nearDensities[size] = 0;
	size++;
	return true;
}

size_t Particles::getSize() {
	return size;
}

glm::vec2 Particles::getPosition(size_t index) {
	if (index >= size) {
		return glm::vec2(0, 0);
	}

	return glm::vec2(positions_x[index], positions_y[index]);
}

glm::vec2 Particles::getPredictedPosition(size_t index) {
	if (index >= size) {
		return glm::vec2(0, 0);
	}

	return glm::vec2(predictedPositions_x[index], predictedPositions_y[index]);
}

void Particles::setVelocity(size_t index, glm::vec2 velocity) {
	if (index >= size) {
		return;
	}

	velocities_x[index] = velocity.x;
	velocities_y[index] = velocity.y;
}

void Particles::setPosition(size_t index, glm::vec2& position) {
	if (index >= size) {
		return;
	}

	positions_x[index] = position.x;
	positions_y[index] = position.y;
}

void Particles::addForce(size_t index, glm::vec2 force) {
	if (index >= size) {
		return;
	}

	forces_x[index] += force.x;
	forces_y[index] += force.y;
}

void Particles::update(const float deltaTime) {
	for (int index = 0; index < size; index++) {
		velocities_x[index] += forces_x[index] * deltaTime;
		velocities_y[index] += forces_y[index] * deltaTime;

		forces_x[index] = 0.0f;
		forces_y[index] = 0.0f;
		densities[index] = 0.0f;
		nearDensities[index] = 0.0f;

		positions_x[index] += velocities_x[index] * deltaTime;
		positions_y[index] += velocities_y[index] * deltaTime;
	}
}

void Particles::updatePredicted(const float deltaTime) {
	for (int index = 0; index < size; index++) {
		velocities_x[index] += forces_x[index] * deltaTime;
		velocities_y[index] += forces_y[index] * deltaTime;
		forces_x[index] = 0.0f;
		forces_y[index] = 0.0f;
		predictedPositions_x[index] = positions_x[index] + velocities_x[index] * deltaTime;
		predictedPositions_y[index] = positions_y[index] + velocities_y[index] * deltaTime;
	}
}

float Particles::calculatePredictedDistance(size_t firstIndex, size_t secondIndex) {
	if (firstIndex >= size || secondIndex >= size) {
		return 0.0f;
	}

	return sqrt(pow(predictedPositions_x[firstIndex] - predictedPositions_x[secondIndex], 2) +
		pow(predictedPositions_y[firstIndex] - predictedPositions_y[secondIndex], 2));
}

void Particles::addDensity(size_t index, float density) {
	densities[index] += density;
}

void Particles::addNearDensity(size_t index, float nearDensity) {
	nearDensities[index] += nearDensity;
}

float Particles::getDensity(size_t index) {
	if (index >= size) {
		return 0.0f;
	}

	return densities[index];
}

float Particles::getNearDensity(size_t index) {
	if (index >= size) {
		return 0.0f;
	}

	return nearDensities[index];
}

glm::vec2 Particles::getVelocity(size_t index) {
	if (index >= size) {
		return glm::vec2(0.0f, 0.0f);
	}

	return glm::vec2(velocities_x[index], velocities_y[index]);
}
