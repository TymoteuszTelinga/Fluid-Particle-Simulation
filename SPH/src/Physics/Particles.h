#pragma once

#include <vector>
#include <glm/glm.hpp>

class Particles
{
public:
	Particles(size_t capacity);

	~Particles();

	bool addParticle(float x_pos, float y_pos);
	size_t getSize();

	glm::vec2 getPosition(size_t index);
	glm::vec2 getPredictedPosition(size_t index);
	void setVelocity(size_t index, glm::vec2 velocity);
	void setPosition(size_t index, glm::vec2& position);
	void addForce(size_t index, glm::vec2 force);
	void update(const float deltaTime);
	void updatePredicted(const float deltaTime);
	float calculatePredictedDistance(size_t firstIndex, size_t secondIndex);
	void addDensity(size_t index, float density);
	void addNearDensity(size_t index, float nearDensity);
	float getDensity(size_t index);
	float getNearDensity(size_t index);
	glm::vec2 getVelocity(size_t index);

	void sendToCuda();
	void getFromCuda();
	void updatePredictedCuda(float deltaTime);
	void getFromCudaBeforeSpatial();
		

public:
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

public:
	float* c_positions_x;
	float* c_positions_y;

	float* c_predictedPositions_x;
	float* c_predictedPositions_y;

	float* c_velocities_x;
	float* c_velocities_y;

	float* c_forces_x;
	float* c_forces_y;

	float* c_densities;
	float* c_nearDensities;

	int* c_lookup_index;
	int* c_lookup_key;
	int* c_indices;

	int c_indices_size;
};

