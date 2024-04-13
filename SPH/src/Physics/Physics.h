#pragma once

#include <vector>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Kernel.h"
#include "Physics/Particle.h"

#include "Physics/Properties/Density.h"
#include "Physics/Forces/Gravity.h"
#include "Physics/Forces/Pressure.h"
#include "Physics/Forces/Viscosity.h"
#include "Physics/Forces/CollisionHandler.h"
#include "Physics/NeighbourSearch.h"

class Physics
{
public:
	Physics(PhysicsSpecification& spec): p_spec(spec) {
		l_Gravity = CreateScope<Gravity>(spec);
		l_Density = CreateScope<Density>(spec);
		l_Pressure = CreateScope<Pressure>(spec);
		l_Viscosity = CreateScope<Viscosity>(spec);
		l_CollisionHandler = CreateScope<CollisionHandler>(spec);
		l_NeighbourSearch = CreateScope<NeighbourSearch>(spec);
	}

	void Apply(std::vector<Particle>& particles, const float deltaTime) const;

	PhysicsSpecification& getSpecification() {
		return p_spec;
	}

private:
	void Update(std::vector<Particle>& particles, const float deltaTime) const;
	void PredictParticlePositions(std::vector<Particle>& particles, const float deltaTime) const;

private:
	PhysicsSpecification& p_spec;
	Scope<Gravity> l_Gravity;
	Scope<Pressure> l_Pressure;
	Scope<Density> l_Density;
	Scope<Viscosity> l_Viscosity;
	Scope<CollisionHandler> l_CollisionHandler;
	Scope<NeighbourSearch> l_NeighbourSearch;
};

