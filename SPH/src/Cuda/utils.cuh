#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ size_t PositionToCellKey(float* positions_x, float* positions_y, size_t index, float kernelRange, int cellRows, int cellCols);

__device__ float calculateSquareDistance(float* positions_x, float* positions_y, int fIdx, int sIdx);

__device__ float calculatePressure(float density, float gasConstant, float restDensity);

__device__ float calculateNearPressure(float nearDensity, float nearPressureCoef);