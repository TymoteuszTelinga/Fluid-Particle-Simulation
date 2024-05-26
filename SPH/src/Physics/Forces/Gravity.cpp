#include "Gravity.h"

#include "Cuda/Kernels.h"


void Gravity::Apply(Ref<Particles> particles, float deltaTime) const {
	GravityCuda(-p_spec.GravityAcceleration, particles->getSize(), deltaTime);
}
