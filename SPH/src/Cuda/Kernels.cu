#include "Kernels.cuh"

#include "stdio.h"

__constant__ int offset_y[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1};
__constant__ int offset_x[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1};

__device__ size_t PositionToCellKey(float pos_x, float pos_y, float kernelRange, int cellRows, int cellCols) {
	int cellX = (int)(pos_x / kernelRange);
	int cellY = (int)(pos_y / kernelRange);

	if (cellRows - 1 < cellX) {
		cellX = cellRows - 1;
	}
	else if (0 > cellX) {
		cellX = 0;
	}

	if (cellCols - 1 < cellY) {
		cellY = cellRows - 1;
	}
	else if (0 > cellY) {
		cellY = 0;
	}

	return cellX + cellY * cellRows;
}

__device__ float calculateDistance(float f_pos_x, float f_pos_y, float s_pos_x, float s_pos_y) {
	float x_diff = f_pos_x - s_pos_x;
	float y_diff = f_pos_y - s_pos_y;
	return sqrt(x_diff * x_diff + y_diff * y_diff);
}

__device__ float calculatePressure(float density, float gasConstant, float restDensity) {
	return gasConstant * (density - restDensity);
}

__device__ float calculateNearPressure(float nearDensity, float nearPressureCoef) {
	return nearDensity * nearPressureCoef;
}

__device__ float poly6Kernel(float distance, float radius, float factor) {
	if (distance <= radius) {
		float diff = radius * radius - distance * distance;
		return factor * pow(diff, 3);
	}
	return 0;
}


__device__ float spiky2Kernel(float distance, float radius, float factor) {
	if (distance < radius) {
		float diff = radius - distance;
		return factor * pow(diff, 2);
	}
	return 0;
}

__device__ float spiky3Kernel(float distance, float radius, float factor) {
	if (distance < radius) {
		float diff = radius - distance;
		return factor * pow(diff, 3);
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
		return factor * pow(diff, 2);
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

__global__ void GravityKernel(float* forces_y, float acc, size_t size) {
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		forces_y[idx] = acc;
	}
}

__global__ void UpdatePredictedKernel(float* velocities_x, float* velocities_y, float* forces_x, float* forces_y,
	float* predictedPos_x, float* predictedPos_y, float* pos_x, float* pos_y, float deltaTime, size_t size) {
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		velocities_x[idx] += forces_x[idx] * deltaTime;
		velocities_y[idx] += forces_y[idx] * deltaTime;
		forces_x[idx] = 0.0f;
		forces_y[idx] = 0.0f;
		predictedPos_x[idx] = pos_x[idx] + velocities_x[idx] * deltaTime;
		predictedPos_y[idx] = pos_y[idx] + velocities_y[idx] * deltaTime;
	}
}



__global__ void DensityKernel(float* pred_pos_x, float* pred_pos_y, float* density, float* nearDensity,  float kernelRange,
		float densityFactor, float nearDensityFactor, size_t size, int* lookupIndex, int* lookupKey, int cellRows, int cellColls, int* indices, size_t indices_size) {
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}

	float temp_density = 0;
	float temp_nearDensity = 0;
	float sqrRange = kernelRange * kernelRange;
	size_t cellKey = PositionToCellKey(pred_pos_x[idx], pred_pos_y[idx], kernelRange, cellRows, cellColls);
	for (int i = 0; i < 9; i++) {
		int neighbourKey = cellKey + offset_y[i] * cellRows + offset_x[i];
		if (neighbourKey < 0 || neighbourKey >= indices_size) {
			continue;
		}

		size_t startIndice = indices[neighbourKey];
		for (size_t lIdx = startIndice; lIdx < size; lIdx++) {
			if (lookupKey[lIdx] != neighbourKey) {
				break;
			}
			
			size_t pIdx = lookupIndex[lIdx];

			float distance = calculateDistance(pred_pos_x[idx], pred_pos_y[idx], pred_pos_x[pIdx], pred_pos_y[pIdx]);
			if (distance * distance > sqrRange) {
				continue;
			}

			temp_density += spiky2Kernel(distance, kernelRange, densityFactor);
			temp_nearDensity += spiky3Kernel(distance, kernelRange, nearDensityFactor);
		}
	}
	density[idx] = temp_density;
	nearDensity[idx] = temp_nearDensity;
}

__global__ void PressureKernel(float* pred_pos_x, float* pred_pos_y, float* forces_x, float* forces_y,  float* densities, float* nearDensities, float kernelRange, 
	float spiky2DerivFactor, float spiky3DerivFactor, float gasConstant, float restDensity, float nearPressureCoef, size_t size,
	int* lookupIndex, int* lookupKey, int cellRows, int cellColls, int* indices, size_t indices_size) {
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	float temp_force_x = 0;
	float temp_force_y = 0;
	float density = densities[idx];
	float nearDensity = nearDensities[idx];

	float pressure = calculatePressure(density, gasConstant, restDensity);
	float nearPressure = calculateNearPressure(nearDensity, nearPressureCoef);

	float sqrRange = kernelRange * kernelRange;
	size_t cellKey = PositionToCellKey(pred_pos_x[idx], pred_pos_y[idx], kernelRange, cellRows, cellColls);
	for (int i = 0; i < 9; i++) {
		int neighbourKey = cellKey + offset_y[i] * cellRows + offset_x[i];
		if (neighbourKey < 0 || neighbourKey >= indices_size) {
			continue;
		}

		size_t startIndice = indices[neighbourKey];
		for (size_t lIdx = startIndice; lIdx < size; lIdx++) {
			if (lookupKey[lIdx] != neighbourKey) {
				break;
			}

			size_t pIdx = lookupIndex[lIdx];

			if (pIdx == idx) {
				continue;
			}

			float distance = calculateDistance(pred_pos_x[idx], pred_pos_y[idx], pred_pos_x[pIdx], pred_pos_y[pIdx]);
			if (distance * distance > sqrRange) {
				continue;
			}

			float slope = spiky2DerivKernel(distance, kernelRange, spiky2DerivFactor);
			float nearSlope = spiky3DerivKernel(distance, kernelRange, spiky3DerivFactor);
			if (slope == 0) {
				continue;
			}

			float pDensity = densities[pIdx];
			float pNearDensity = nearDensities[pIdx];

			float pPressure = calculatePressure(pDensity, gasConstant, restDensity);
			float pNearPressure = calculateNearPressure(pNearDensity, nearPressureCoef);
			
			float dir_x = 0.0f;
			float dir_y = 1.0f;
			if (distance > 0.0f) {
				dir_x = (pred_pos_x[pIdx] - pred_pos_x[idx]) / distance;
				dir_y = (pred_pos_y[pIdx] - pred_pos_y[idx]) / distance;
			}


			float pressureCoef = 0.5f * (pressure + pPressure) * slope / (density * pDensity);
			float nearPressureCoef = 0.5f * (nearPressure + pNearPressure) * nearSlope / (density * pNearDensity);

			temp_force_x += (pressureCoef + nearPressureCoef) * dir_x;
			temp_force_y += (pressureCoef + nearPressureCoef) * dir_y;
		}
	}
	forces_x[idx] += temp_force_x;
	forces_y[idx] += temp_force_y;
}

void GravityCuda(float* forces_y, float acc, size_t size) {
	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}
	
	GravityKernel <<<blocks, TPB >> > (forces_y, acc, size);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "gravity launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching gravity!\n", cudaStatus);
	}
}


void UpdatePredictedFromCuda(float* velocities_x, float* velocities_y, float* forces_x, float* forces_y,
	float* predictedPos_x, float* predictedPos_y, float* pos_x, float* pos_y, float deltaTime, size_t size) {
	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}

	UpdatePredictedKernel<<<blocks, TPB>>> (velocities_x, velocities_y, forces_x, forces_y, predictedPos_x,
		predictedPos_y, pos_x, pos_y, deltaTime, size);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "update launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching update!\n", cudaStatus);
	}
}


void DensityCuda(float* pred_pos_x, float* pred_pos_y, float* density, float* nearDensity, float kernelRange,
	float densityFactor, float nearDensityFactor, size_t size, int* lookupIndex, int* lookupKey, int cellRows, int cellColls, int* indices, size_t indices_size) {

	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}

	DensityKernel << <blocks, TPB >> > (pred_pos_x, pred_pos_y, density, nearDensity, kernelRange, densityFactor,
		nearDensityFactor, size, lookupIndex, lookupKey, cellRows, cellColls, indices, indices_size);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "density launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching density!\n", cudaStatus);
	}
}

void PressureCuda(float* pred_pos_x, float* pred_pos_y, float* forces_x, float* forces_y, float* densities, float* nearDensities, float kernelRange,
	float spiky2DerivFactor, float spiky3DerivFactor, float gasConstant, float restDensity, float nearPressureCoef, size_t size,
	int* lookupIndex, int* lookupKey, int cellRows, int cellColls, int* indices, size_t indices_size) {

	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}

	PressureKernel << <blocks, TPB >> > (pred_pos_x, pred_pos_y, forces_x, forces_y, densities, nearDensities, kernelRange,
		spiky2DerivFactor, spiky3DerivFactor, gasConstant, restDensity, nearPressureCoef, size,
		lookupIndex, lookupKey, cellRows, cellColls, indices, indices_size);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "pressure launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pressure!\n", cudaStatus);
	}
}

