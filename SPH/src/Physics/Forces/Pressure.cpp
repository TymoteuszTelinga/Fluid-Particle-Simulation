#include "Pressure.h"

#include <thread>

#include "Cuda/Kernels.h"
#include "Core/Base.h"


void Pressure::Apply(Ref<Particles> particles, float deltaTime) const {
	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	int cellCols = (int)(p_spec.Height / p_spec.KernelRange) + 1;

	PressureCuda(p_spec.KernelRange, kernel->Spiky2DerivFactor, kernel->Spiky3DerivFactor, p_spec.GasConstant, p_spec.RestDensity,
		p_spec.NearPressureCoef, particles->getSize(), cellRows, cellCols, particles->c_indices_addr, particles->c_indices_size, deltaTime);
}


