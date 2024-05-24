#include "Physics.h"


void Physics::Apply(Ref<Particles> particles, const float deltaTime) const {
	m_Kernel->updateFactors(p_spec.KernelRange);

	l_Gravity->Apply(particles);

	particles->updatePredicted(1.0f / 120.0f);

	l_NeighbourSearch->UpdateSpatialLookup(particles);
	l_Density->Calculate(particles);
	l_Pressure->Apply(particles);
	l_Viscosity->Apply(particles);

	particles->update(deltaTime);
	l_CollisionHandler->Resolve(particles);
}

void Physics::ApplyCuda(Ref<Particles> particles, const float deltaTime) const {
	m_Kernel->updateFactors(p_spec.KernelRange);
	particles->sendToCuda();

	l_Gravity->ApplyCuda(particles);
	particles->updatePredictedCuda(1.0f / 120.0f);

	particles->getFromCuda();

	l_NeighbourSearch->UpdateSpatialLookup(particles);

	l_Density->CalculateCuda(particles);

	l_Pressure->ApplyCuda(particles);
	particles->getFromCuda();
	l_Viscosity->Apply(particles);


	particles->update(deltaTime);
	l_CollisionHandler->Resolve(particles);
	


}
