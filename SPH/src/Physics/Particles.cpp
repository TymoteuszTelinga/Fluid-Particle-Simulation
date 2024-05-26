#include "Particles.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Cuda/Kernels.h"
#include "iostream"


Particles::Particles(size_t capacity) : capacity(capacity), size(0)
{
	positions_x = new float[capacity];
	positions_y = new float[capacity];

	predictedPositions_x = new float[capacity];
	predictedPositions_y = new float[capacity];

	CUDA_CALL(cudaGetSymbolAddress((void**)&c_positions_x_addr, c_positions_x));
	CUDA_CALL(cudaGetSymbolAddress((void**)&c_positions_y_addr, c_positions_y));
	CUDA_CALL(cudaGetSymbolAddress((void**)&c_pred_positions_x_addr, c_predicted_positions_x));
	CUDA_CALL(cudaGetSymbolAddress((void**)&c_pred_positions_y_addr, c_predicted_positions_y));
	CUDA_CALL(cudaGetSymbolAddress((void**)&c_lookup_indexes_addr, c_lookup_indexes));
	CUDA_CALL(cudaGetSymbolAddress((void**)&c_lookup_keys_addr, c_lookup_keys));
	CUDA_CALL(cudaMalloc(&c_indices_addr, capacity * sizeof(int)));
	c_indices_size = capacity;
}

Particles::~Particles() {
	delete[] positions_x;
	delete[] positions_y;

	delete[] predictedPositions_x;
	delete[] predictedPositions_y;

	CUDA_CALL(cudaFree(c_indices_addr));
}

void Particles::sendToCuda() {
	CUDA_CALL(cudaMemcpy(c_positions_x_addr, positions_x, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(c_positions_y_addr, positions_y, size * sizeof(float), cudaMemcpyHostToDevice));
}

void Particles::getFromCudaBeforeSpatial() {
	CUDA_CALL(cudaMemcpy(predictedPositions_x, c_pred_positions_x_addr, size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(predictedPositions_y, c_pred_positions_y_addr, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void Particles::getFromCuda() {
	CUDA_CALL(cudaMemcpy(positions_x, c_positions_x_addr, size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(positions_y, c_positions_y_addr, size * sizeof(float), cudaMemcpyDeviceToHost));
}


bool Particles::addParticle(float x_pos, float y_pos) {
	if (size >= capacity) {
		return false;
	}

	positions_x[size] = x_pos;
	positions_y[size] = y_pos;
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








float Particles::calculatePredictedDistance(size_t firstIndex, size_t secondIndex) {
	if (firstIndex >= size || secondIndex >= size) {
		return 0.0f;
	}

	return sqrt(pow(predictedPositions_x[firstIndex] - predictedPositions_x[secondIndex], 2) +
		pow(predictedPositions_y[firstIndex] - predictedPositions_y[secondIndex], 2));
}