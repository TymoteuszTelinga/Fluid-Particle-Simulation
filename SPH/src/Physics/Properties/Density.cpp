#include "Density.h"

#include "Cuda/Kernels.h"

void Density::Calculate(Ref<Particles> particles) const {
	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	int cellCols = (int)(p_spec.Height / p_spec.KernelRange) + 1;

	DensityCuda(p_spec.KernelRange, kernel->Spiky2Factor, kernel->Spiky3Factor,
		particles->getSize(), cellRows, cellCols, particles->c_indices_addr, particles->c_indices_size);
}