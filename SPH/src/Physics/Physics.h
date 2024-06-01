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


/**
* Facade class serving as entry point for physics related operations
*/
class Physics
{
public:
	/**
	* The main and the only one constructor.
	* 
	* 
	* @param spec physics related specification, it is spread further to dependent classes as a reference
	* @param obstacles collection of static obstacles inside the simulation environment, they have influence on particles by collision
	* @param in area serving for particles as entry to environment
	* @param out area serving for particles as exit from environment
	*/
	Physics(physicsSpecification& spec, std::vector<obstacle>& obstacles, flow_area in, flow_area out): p_spec(spec) {
		m_Kernel = CreateRef<KernelFactors>();
		l_NeighbourSearch = CreateScope<NeighbourSearch>(p_spec);
		l_Gravity = CreateScope<Gravity>(p_spec);
		l_Density = CreateScope<Density>(p_spec, m_Kernel);
		l_Pressure = CreateScope<Pressure>(p_spec, m_Kernel);
		l_Viscosity = CreateScope<Viscosity>(p_spec, m_Kernel);
		l_CollisionHandler = CreateScope<CollisionHandler>(p_spec, obstacles);
		l_Flow = CreateScope<Flow>(in, out);
	}

	/**
	* Applies the chain of operations on passed particles in the order below:
	* 1. Generates new particles if there is still capacity for them 
	* 2. Applies gravity force on particles and calculates future position 
	* 3. Orders particles into cells and prepare efficient spatial lookup 
	* 4. Calculates particles' density and near density 
	* 5. Applies pressure force to particles
	* 6. Applies viscosity force to particles
	* 7. Resolves collision with simulation boundaries and defined obstacles
	* 8. Removes particles located in exit area of simulation
	* 
	* @param particles particles which should be affected by physics.
	* @param deltaTime time from the last physic influence. It is used for velocity and position calculations. 
	* @param fillSize amount of particles which should be generated.
	*/
	void Apply(Ref<Particles> particles, const float deltaTime, const size_t fillSize) const;

	/**
	* returs reference to currently used physic specification. Altering its attributes will change
	* behaviour of the simulation.
	* 
	* @return reference of physics specification.
	*/
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
	Scope<NeighbourSearch> l_NeighbourSearch;
	Ref<KernelFactors> m_Kernel;
};

