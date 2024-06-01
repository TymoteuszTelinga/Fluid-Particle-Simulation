#pragma once

#include <vector>

#include "Core/Base.h"

#include "Physics/physicsSpecification.h"

#include "Physics/Entities/Particles.h"

#include "Physics/Forces/Density.h"
#include "Physics/Forces/Gravity.h"
#include "Physics/Forces/Pressure.h"
#include "Physics/Forces/Viscosity.h"

#include "Physics/Logic/NeighbourSearch.h"
#include "Physics/Logic/KernelFactors.h"
#include "Physics/Logic/CollisionHandler.h"
#include "Physics/Logic/Flow.h"

class Physics
{
public:
	Physics(physicsSpecification& spec, std::vector<obstacle>& obstacles, flow_area in, flow_area out): p_spec(spec) {
		m_Kernel = CreateRef<KernelFactors>();
		l_NeighbourSearch = CreateRef<NeighbourSearch>(p_spec);
		l_Gravity = CreateScope<Gravity>(p_spec);
		l_Density = CreateScope<Density>(p_spec, l_NeighbourSearch, m_Kernel);
		l_Pressure = CreateScope<Pressure>(p_spec, l_NeighbourSearch, m_Kernel);
		l_Viscosity = CreateScope<Viscosity>(p_spec, l_NeighbourSearch, m_Kernel);
		l_CollisionHandler = CreateScope<CollisionHandler>(p_spec, obstacles);
		l_Flow = CreateScope<Flow>(in, out);
	}

	void Apply(Ref<Particles> particles, const float deltaTime, const size_t fillSize) const;

	physicsSpecification& getSpecification() {
		return p_spec;
	}

private:
	physicsSpecification p_spec;
	Scope<Gravity> l_Gravity;
	Scope<Pressure> l_Pressure;
	Scope<Density> l_Density;
	Scope<Viscosity> l_Viscosity;
	Scope<CollisionHandler> l_CollisionHandler;
	Scope<Flow> l_Flow;
	Ref<NeighbourSearch> l_NeighbourSearch;
	Ref<KernelFactors> m_Kernel;
};

