#include "KernelFactors.h"


void KernelFactors::updateFactors(float kernelRadius) {
	Poly6Factor = KernelFactors::Poly6ConstFactor / pow(kernelRadius, 8);
	Spiky3Factor = KernelFactors::Spiky3ConstFactor / pow(kernelRadius, 5);
	Spiky2Factor = KernelFactors::Spiky2ConstFactor / pow(kernelRadius, 4);
	Poly6DerivFactor = KernelFactors::Poly6DerivConstFactor / pow(kernelRadius, 8);
	Spiky3DerivFactor = -KernelFactors::Spiky3DerivConstFactor / pow(kernelRadius, 5);
	Spiky2DerivFactor = -KernelFactors::Spiky2DerivConstFactor / pow(kernelRadius, 4);
	ViscosityLaplacianFactor = KernelFactors::ViscosityLaplacianConstFactor / pow(kernelRadius, 5);
}
