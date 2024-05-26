#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float poly6Kernel(float distance, float radius, float factor);

__device__ float spiky2Kernel(float distance, float radius, float factor);

__device__ float spiky3Kernel(float distance, float radius, float factor);

__device__ float poly6DerivKernel(float distance, float radius, float factor);

__device__ float spiky3DerivKernel(float distance, float radius, float factor);

__device__ float spiky2DerivKernel(float distance, float radius, float factor);