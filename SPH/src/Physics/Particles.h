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
	size_t getCapacity();

	glm::vec2 getPosition(size_t index);
	glm::vec2 getPredictedPosition(size_t index);
	float calculatePredictedDistance(size_t firstIndex, size_t secondIndex);

	void remove(size_t index);

	void sendToCuda();
	void getFromCuda();
	void getFromCudaBeforeSpatial();
		

private:
	const size_t capacity;
	size_t size;

	float* positions_x;
	float* positions_y;

	float* velocities_x;
	float* velocities_y;

	float* predictedPositions_x;
	float* predictedPositions_y;

public:
	float* c_positions_x_addr;
	float* c_positions_y_addr;
	float* c_velocities_x_addr;
	float* c_velocities_y_addr;
	float* c_pred_positions_x_addr;
	float* c_pred_positions_y_addr;
	int* c_lookup_indexes_addr;
	int* c_lookup_keys_addr;
	int* c_indices_addr;
	int c_indices_size;

};

