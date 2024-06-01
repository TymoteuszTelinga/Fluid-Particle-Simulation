#include "utils.cuh"

__device__ size_t PositionToCellKey(float* positions_x, float* positions_y, size_t index, float kernelRange, int cellRows, int cellCols) {
	int cellX = (int)(positions_x[index] / kernelRange);
	int cellY = (int)(positions_y[index] / kernelRange);

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

__device__ float calculateSquareDistance(float* positions_x, float* positions_y, int fIdx, int sIdx) {
	float x_diff = positions_x[fIdx] - positions_x[sIdx];
	float y_diff = positions_y[fIdx] - positions_y[sIdx];
	return x_diff * x_diff + y_diff * y_diff;
}

__device__ float calculatePressure(float density, float gasConstant, float restDensity) {
	return gasConstant * (density - restDensity);
}

__device__ float calculateNearPressure(float nearDensity, float nearPressureCoef) {
	return nearDensity * nearPressureCoef;
}