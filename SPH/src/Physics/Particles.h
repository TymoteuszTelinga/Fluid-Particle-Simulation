#pragma once

#include <vector>
#include <glm/glm.hpp>

class Particles
{
public:
	Particles(size_t capacity) : capacity(capacity), size(0) 
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
	}

	~Particles() {
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
	}

	bool addParticle(float x_pos, float y_pos) {
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

	size_t getSize() {
		return size;
	}

	glm::vec2 getPosition(size_t index) {
		if (index >= size) {
			return glm::vec2(0, 0);
		}

		return glm::vec2(positions_x[index], positions_y[index]);
	}

	glm::vec2 getPredictedPosition(size_t index) {
		if (index >= size) {
			return glm::vec2(0, 0);
		}

		return glm::vec2(predictedPositions_x[index], predictedPositions_y[index]);
	}

	void setVelocity(size_t index, glm::vec2 velocity) {
		if (index >= size) {
			return;
		}

		velocities_x[index] = velocity.x;
		velocities_y[index] = velocity.y;
	}

	void setPosition(size_t index, glm::vec2& position) {
		if (index >= size) {
			return;
		}

		positions_x[index] = position.x;
		positions_y[index] = position.y;
	}

	void addForce(size_t index, glm::vec2 force) {
		if (index >= size) {
			return;
		}

		forces_x[index] += force.x;
		forces_y[index] += force.y;
	}

	void update(const float deltaTime) {
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

	void updatePredicted(const float deltaTime) {
		for (int index = 0; index < size; index++) {
			velocities_x[index] += forces_x[index] * deltaTime;
			velocities_y[index] += forces_y[index] * deltaTime;
			forces_x[index] = 0.0f;
			forces_y[index] = 0.0f;
			predictedPositions_x[index] = positions_x[index] + velocities_x[index] * deltaTime;
			predictedPositions_y[index] = positions_y[index] + velocities_y[index] * deltaTime;
		}
	}

	float calculatePredictedDistance(size_t firstIndex, size_t secondIndex) {
		if (firstIndex >= size || secondIndex >= size) {
			return 0.0f;
		}

		return sqrt(pow(predictedPositions_x[firstIndex] - predictedPositions_x[secondIndex], 2) +
			pow(predictedPositions_y[firstIndex] - predictedPositions_y[secondIndex], 2));
	}

	void addDensity(size_t index, float density) {
		densities[index] += density;
	}

	void addNearDensity(size_t index, float nearDensity) {
		nearDensities[index] += nearDensity;
	}

	float getDensity(size_t index) {
		if (index >= size) {
			return 0.0f;
		}

		return densities[index];
	}

	float getNearDensity(size_t index) {
		if (index >= size) {
			return 0.0f;
		}

		return nearDensities[index];
	}

	glm::vec2 getVelocity(size_t index) {
		if (index >= size) {
			return glm::vec2(0.0f, 0.0f);
		}

		return glm::vec2(velocities_x[index], velocities_y[index]);
	}

private:
	const size_t capacity;
	size_t size;

	float* positions_x;
	float* positions_y;

	float* predictedPositions_x;
	float* predictedPositions_y;

	float* velocities_x;
	float* velocities_y;

	float* forces_x;
	float* forces_y;

	float* densities;
	float* nearDensities;
};

