#include "Viscosity.h"

#include "Cuda/Kernels.h"

void Viscosity::Apply(Ref<Particles> particles, float deltaTime) {
	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	int cellCols = (int)(p_spec.Height / p_spec.KernelRange) + 1;

	ViscosityCuda(p_spec.KernelRange, kernel->Poly6Factor, p_spec.ViscosityStrength, particles->getSize(), cellRows, cellCols, particles->c_indices_addr, particles->c_indices_size, deltaTime);
}
