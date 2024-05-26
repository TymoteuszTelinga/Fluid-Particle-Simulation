#include "Cuda/smoothingKernels.cuh"


__device__ float poly6Kernel(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff = radius * radius - distance * distance;
		return factor * diff * diff * diff;
	}
	return 0;
}


__device__ float spiky2Kernel(float distance, float radius, float factor) {
	if (distance < radius) {
		float diff = radius - distance;
		return factor * diff * diff;
	}
	return 0;
}

__device__ float spiky3Kernel(float distance, float radius, float factor) {
	if (distance < radius) {
		float diff = radius - distance;
		return factor * diff * diff * diff;
	}
	return 0;
}

__device__ float poly6DerivKernel(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff = radius * radius - distance * distance;
		return factor * distance * diff * diff;
	}
	return 0;
}

__device__ float spiky3DerivKernel(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff = radius - distance;
		return factor * diff * diff;
	}
	return 0;
}

__device__ float spiky2DerivKernel(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff = radius - distance;
		return factor * diff;
	}
	return 0;
}