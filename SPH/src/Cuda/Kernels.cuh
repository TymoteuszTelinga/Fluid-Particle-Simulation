#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const size_t TPB = 1024;

void GravityCuda(float* forces_y, float acc, size_t size);

void UpdatePredictedFromCuda(float* velocities_x, float* velocities_y, float* forces_x, float* forces_y,
	float* predictedPos_x, float* predictedPos_y, float* pos_x, float* pos_y, float deltaTime, size_t size);

void DensityCuda(float* pred_pos_x, float* pred_pos_y, float* density, float* nearDensity, float kernelRange,
	float densityFactor, float nearDensityFactor, size_t size, int* lookupIndex, int* lookupKey, int cellRows, int cellColls, int* indices, size_t indices_size);

void PressureCuda(float* pred_pos_x, float* pred_pos_y, float* forces_x, float* forces_y, float* densities, float* nearDensities, float kernelRange,
	float spiky2DerivFactor, float spiky3DerivFactor, float gasConstant, float restDensity, float nearPressureCoef, size_t size,
	int* lookupIndex, int* lookupKey, int cellRows, int cellColls, int* indices, size_t indices_size);