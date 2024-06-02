#include "Physics.h"


void Physics::Apply(Ref<Particles> particles, const float deltaTime) const {
	m_Kernel->updateFactors(p_spec.KernelRange);
	l_Flow->in(std::max(1,(int)(p_spec.ParticlesPerSecond * deltaTime)), particles);
	particles->sendToCuda();

	l_Gravity->Apply(particles, deltaTime);
	particles->getFromCudaBeforeSpatial();

	l_NeighbourSearch->UpdateSpatialLookup(particles);
	l_Density->Calculate(particles);
	l_Pressure->Apply(particles, deltaTime);
	l_Viscosity->Apply(particles, deltaTime);
	
	l_CollisionHandler->Resolve(particles, deltaTime);
	particles->getFromCuda();
	l_Flow->out(particles);
}
