#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Core/ComonTypes.h"

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

const size_t TPB = 512;
const size_t PARTICLES_LIMIT = 2000;

extern __device__ float c_positions_x[PARTICLES_LIMIT];
extern __device__ float c_positions_y[PARTICLES_LIMIT];
extern __device__ float c_predicted_positions_x[PARTICLES_LIMIT];
extern __device__ float c_predicted_positions_y[PARTICLES_LIMIT];
extern __device__ float c_velocities_x[PARTICLES_LIMIT];
extern __device__ float c_velocities_y[PARTICLES_LIMIT];
extern __device__ float c_densities[PARTICLES_LIMIT];
extern __device__ float c_near_densities[PARTICLES_LIMIT];
extern __device__ int c_lookup_indexes[PARTICLES_LIMIT];
extern __device__ int c_lookup_keys[PARTICLES_LIMIT];

void GravityCuda(float acc, size_t size, float deltaTime);

void DensityCuda(float kernelRange, float densityFactor, float nearDensityFactor, size_t size, int cellRows, int cellColls, int* indices, size_t indices_size);

void PressureCuda(float kernelRange, float spiky2DerivFactor, float spiky3DerivFactor, float gasConstant, float restDensity,
	float nearPressureCoef, size_t size, int cellRows, int cellColls, int* indices, size_t indices_size, float deltaTime);

void ViscosityCuda(float kernelRange, float poly6Factor, float viscosityStrength, size_t size, int cellRows,
	int cellColls, int* indices, size_t indices_size, float deltaTime);

void UpdateAndCollisionCuda(size_t size, float deltaTime, float collisionDamping, float particlesRadius, float min_x, float max_x, 
	float min_y, float max_y, obstacle* obstacles, size_t obstacles_size);
