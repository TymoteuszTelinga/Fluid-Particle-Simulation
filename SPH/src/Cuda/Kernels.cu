#include "Kernels.h"
#include "Cuda/smoothingKernels.cuh"
#include "Cuda/utils.cuh"

#include "stdio.h"





__device__ float c_positions_x[PARTICLES_LIMIT];
__device__ float c_positions_y[PARTICLES_LIMIT];
__device__ float c_predicted_positions_x[PARTICLES_LIMIT];
__device__ float c_predicted_positions_y[PARTICLES_LIMIT];
__device__ float c_velocities_x[PARTICLES_LIMIT];
__device__ float c_velocities_y[PARTICLES_LIMIT];
				 
__device__ float c_densities[PARTICLES_LIMIT];
__device__ float c_near_densities[PARTICLES_LIMIT];
				 
__device__ int c_lookup_indexes[PARTICLES_LIMIT];
__device__ int c_lookup_keys[PARTICLES_LIMIT];

__constant__ int offset_y[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1};
__constant__ int offset_x[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1};



__global__ void GravityKernel(float acc, size_t size, float deltaTime) {
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}

	c_velocities_y[idx] += acc * deltaTime;
	c_predicted_positions_x[idx] = c_positions_x[idx] + c_velocities_x[idx] * deltaTime;
	c_predicted_positions_y[idx] = c_positions_y[idx] + c_velocities_y[idx] * deltaTime;
}

__global__ void DensityKernel(float kernelRange, float densityFactor, float nearDensityFactor, size_t size, int cellRows, int cellColls, int* indices, size_t indices_size) {
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}

	float temp_density = 0;
	float temp_nearDensity = 0;
	float sqrRange = kernelRange * kernelRange;
	size_t cellKey = PositionToCellKey(c_predicted_positions_x, c_predicted_positions_y, idx, kernelRange, cellRows, cellColls);
	for (int i = 0; i < 9; i++) {
		int neighbourKey = cellKey + offset_y[i] * cellRows + offset_x[i];
		if (neighbourKey < 0 || neighbourKey >= indices_size) {
			continue;
		}

		size_t startIndice = indices[neighbourKey];

		for (size_t lIdx = startIndice; lIdx < size; lIdx++) {
			if (c_lookup_keys[lIdx] != neighbourKey) {
				break;
			}


			size_t pIdx = c_lookup_indexes[lIdx];

			float sqrdistance = calculateSquareDistance(c_predicted_positions_x, c_predicted_positions_y, idx, pIdx);

			if (sqrdistance  > sqrRange) {
				continue;
			}

			float distance = sqrt(sqrdistance);

			temp_density += spiky2Kernel(distance, kernelRange, densityFactor);
			temp_nearDensity += spiky3Kernel(distance, kernelRange, nearDensityFactor);
		}
	}

	c_densities[idx] = temp_density;
	c_near_densities[idx] = temp_nearDensity;
}

__global__ void PressureKernel(float kernelRange, float spiky2DerivFactor, float spiky3DerivFactor, float gasConstant,
	float restDensity, float nearPressureCoef, size_t size, int cellRows, int cellColls, int* indices, size_t indices_size, float deltaTime) {
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}

	float temp_force_x = 0;
	float temp_force_y = 0;
	float density = c_densities[idx];
	float nearDensity = c_near_densities[idx];

	float pressure = calculatePressure(density, gasConstant, restDensity);
	float nearPressure = calculateNearPressure(nearDensity, nearPressureCoef);

	float sqrRange = kernelRange * kernelRange;
	size_t cellKey = PositionToCellKey(c_predicted_positions_x, c_predicted_positions_y, idx, kernelRange, cellRows, cellColls);
	for (int i = 0; i < 9; i++) {
		int neighbourKey = cellKey + offset_y[i] * cellRows + offset_x[i];
		if (neighbourKey < 0 || neighbourKey >= indices_size) {
			continue;
		}

		size_t startIndice = indices[neighbourKey];
		for (size_t lIdx = startIndice; lIdx < size; lIdx++) {
			if (c_lookup_keys[lIdx] != neighbourKey) {
				break;
			}

			size_t pIdx = c_lookup_indexes[lIdx];

			if (pIdx == idx) {
				continue;
			}

			float sqrDistance = calculateSquareDistance(c_predicted_positions_x, c_predicted_positions_y, idx, pIdx);
			if (sqrDistance > sqrRange) {
				continue;
			}
			float distance = sqrt(sqrDistance);

			float slope = spiky2DerivKernel(distance, kernelRange, spiky2DerivFactor);
			float nearSlope = spiky3DerivKernel(distance, kernelRange, spiky3DerivFactor);
			if (slope == 0) {
				continue;
			}

			float pDensity = c_densities[pIdx];
			float pNearDensity = c_near_densities[pIdx];

			float pPressure = calculatePressure(pDensity, gasConstant, restDensity);
			float pNearPressure = calculateNearPressure(pNearDensity, nearPressureCoef);
			
			float dir_x = 0.0f;
			float dir_y = 1.0f;
			if (distance > 0.0f) {
				dir_x = (c_predicted_positions_x[pIdx] - c_predicted_positions_x[idx]) / distance;
				dir_y = (c_predicted_positions_y[pIdx] - c_predicted_positions_y[idx]) / distance;

			}


			float pressureCoef = 0.5f * (pressure + pPressure) * slope / (density * pDensity);
			float nearPressureCoef = 0.5f * (nearPressure + pNearPressure) * nearSlope / (density * pNearDensity);

			temp_force_x += (pressureCoef + nearPressureCoef) * dir_x;
			temp_force_y += (pressureCoef + nearPressureCoef) * dir_y;
		}
	}

	c_velocities_x[idx] += temp_force_x * deltaTime;
	c_velocities_y[idx] += temp_force_y * deltaTime;
}

__global__ void ViscosityKernel(float kernelRange, float poly6Factor, float viscosityStrength,
	size_t size, int cellRows, int cellColls, int* indices, size_t indices_size, float deltaTime) {

	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	float temp_force_x = 0;
	float temp_force_y = 0;

	float velocity_x = c_velocities_x[idx];
	float velocity_y = c_velocities_y[idx];

	float sqrRange = kernelRange * kernelRange;
	size_t cellKey = PositionToCellKey(c_predicted_positions_x, c_predicted_positions_y, idx, kernelRange, cellRows, cellColls);
	for (int i = 0; i < 9; i++) {
		int neighbourKey = cellKey + offset_y[i] * cellRows + offset_x[i];
		if (neighbourKey < 0 || neighbourKey >= indices_size) {
			continue;
		}

		size_t startIndice = indices[neighbourKey];
		for (size_t lIdx = startIndice; lIdx < size; lIdx++) {
			if (c_lookup_keys[lIdx] != neighbourKey) {
				break;
			}

			size_t pIdx = c_lookup_indexes[lIdx];

			if (pIdx == idx) {
				continue;
			}

			float sqrDistance = calculateSquareDistance(c_predicted_positions_x, c_predicted_positions_y, idx, pIdx);
			if (sqrDistance > sqrRange) {
				continue;
			}

			float distance = sqrt(sqrDistance);

			float slope = poly6Kernel(distance, kernelRange, poly6Factor);

			float pVelocity_x = c_velocities_x[pIdx];
			float pVelocity_y = c_velocities_y[pIdx];

			temp_force_x += viscosityStrength * slope * (pVelocity_x - velocity_x);
			temp_force_y += viscosityStrength * slope * (pVelocity_y - velocity_y);
		}
	}

	c_velocities_x[idx] += temp_force_x * deltaTime;
	c_velocities_y[idx] += temp_force_y * deltaTime;
}

__global__ void CollisionKernel(size_t size, float deltaTime, float collisionDamping, float particlesRadius, float min_x, float max_x, float min_y, float max_y, obstacle* obstacles, size_t obstacles_size) {
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	float temp_velocity_x = c_velocities_x[idx];
	float temp_velocity_y = c_velocities_y[idx];
	float last_position_x = c_positions_x[idx];
	float last_position_y = c_positions_y[idx];
	float temp_position_x = last_position_x + temp_velocity_x * deltaTime;
	float temp_position_y = last_position_y +temp_velocity_y * deltaTime;

	bool changed = false;
	if (temp_position_x <= min_x) {
		temp_position_x = min_x + (min_x - temp_position_x);
		temp_velocity_x *= collisionDamping;
		changed = true;
	}
	else if (temp_position_x >= max_x) {
		temp_position_x = max_x + (max_x - temp_position_x);
		temp_velocity_x *= collisionDamping;
		changed = true;
	}

	if (temp_position_y <= min_y) {
		temp_position_y = min_y + (min_y - temp_position_y);
		temp_velocity_y *= collisionDamping;
		changed = true;
	}

	else if (temp_position_y >= max_y) {
		temp_position_y = max_y + (max_y - temp_position_y);
		temp_velocity_y *= collisionDamping;
		changed = true;
	}

	for (size_t oIdx = 0; oIdx < obstacles_size; ++oIdx) {
		obstacle obs = obstacles[oIdx];

		if (last_position_x + particlesRadius + temp_velocity_x * deltaTime >= obs.x_pos &&
			last_position_x + particlesRadius <= obs.x_pos &&
			last_position_y + particlesRadius > obs.y_pos &&
			last_position_y - particlesRadius < obs.y_pos + obs.height) {

			temp_position_x = obs.x_pos + (obs.x_pos - (temp_position_x + particlesRadius)) - particlesRadius;
			temp_velocity_x *= collisionDamping;
			changed = true;
		}
		else if (last_position_x - particlesRadius + temp_velocity_x * deltaTime <= obs.x_pos + obs.width &&
			last_position_x - particlesRadius >= obs.x_pos + obs.width &&
			last_position_y + particlesRadius > obs.y_pos &&
			last_position_y - particlesRadius < obs.y_pos + obs.height) {

			temp_position_x = obs.x_pos + obs.width + (obs.x_pos + obs.width - (temp_position_x - particlesRadius)) + particlesRadius;
			temp_velocity_x *= collisionDamping;
			changed = true;
		}

		if (last_position_x + particlesRadius > obs.x_pos &&
			last_position_x - particlesRadius < obs.x_pos + obs.width &&
			last_position_y + particlesRadius + temp_velocity_y * deltaTime >= obs.y_pos &&
			last_position_y + particlesRadius <= obs.y_pos) {

			temp_position_y = obs.y_pos + (obs.y_pos - (temp_position_y + particlesRadius)) - particlesRadius;
			temp_velocity_y *= collisionDamping;
			changed = true;
		}
		else if (last_position_x + particlesRadius > obs.x_pos &&
			last_position_x - particlesRadius < obs.x_pos + obs.width &&
			last_position_y - particlesRadius + temp_velocity_y * deltaTime <= obs.y_pos + obs.height &&
			last_position_y - particlesRadius >= obs.y_pos + obs.height) {

			temp_position_y = obs.y_pos + obs.height + (obs.y_pos + obs.height - (temp_position_y - particlesRadius)) + particlesRadius;
			temp_velocity_y *= collisionDamping;
			changed = true;
		}
	}


	c_positions_x[idx] = temp_position_x;
	c_positions_y[idx] = temp_position_y;
	if (changed) {
		c_velocities_x[idx] = temp_velocity_x;
		c_velocities_y[idx] = temp_velocity_y;
	}
}

void GravityCuda(float acc, size_t size, float deltaTime) {
	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}
	
	GravityKernel <<<blocks, TPB >> > (acc, size, deltaTime);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "gravity launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching gravity!\n", cudaStatus);
	}
}


void DensityCuda(float kernelRange, float densityFactor, float nearDensityFactor, 
	size_t size, int cellRows, int cellColls, int* indices, size_t indices_size) {

	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}

	DensityKernel << <blocks, TPB >> > (kernelRange, densityFactor, nearDensityFactor, size, cellRows, cellColls, indices, indices_size);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "density launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching density!\n", cudaStatus);
	}
}

void PressureCuda(float kernelRange, float spiky2DerivFactor, float spiky3DerivFactor, float gasConstant,
	float restDensity, float nearPressureCoef, size_t size, int cellRows, int cellColls, int* indices, size_t indices_size, float deltaTime) {
	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}

	PressureKernel <<<blocks, TPB >> > (kernelRange, spiky2DerivFactor, spiky3DerivFactor, gasConstant, restDensity, nearPressureCoef, size, cellRows, cellColls, indices, indices_size, deltaTime);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "pressure launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pressure!\n", cudaStatus);
	}
}

void ViscosityCuda(float kernelRange, float poly6Factor, float viscosityStrength, size_t size, int cellRows, int cellColls, int* indices, size_t indices_size, float deltaTime) {

	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}

	ViscosityKernel <<<blocks, TPB >>> (kernelRange, poly6Factor, viscosityStrength, size, cellRows, cellColls, indices, indices_size, deltaTime);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "viscosity launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching viscosity!\n", cudaStatus);
	}
}

void UpdateAndCollisionCuda(size_t size, float deltaTime, float collisionDamping, float particlesRadius, float min_x, float max_x, float min_y, float max_y, obstacle* obstacles, size_t obstacles_size) {
	size_t blocks = size / TPB;
	if (size % TPB != 0) {
		blocks += 1;
	}

	CollisionKernel<< <blocks, TPB >> > (size, deltaTime, collisionDamping, particlesRadius, min_x, max_x, min_y, max_y, obstacles, obstacles_size);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching collision!\n", cudaStatus);
	}
}

