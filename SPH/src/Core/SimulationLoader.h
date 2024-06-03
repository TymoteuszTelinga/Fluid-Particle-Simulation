#pragma once
#include <string>
#include <vector>

#include "Core/CommonTypes.h"
#include "Physics/PhysicsSpecification.h"

/*
* class that allows to load simulation configuration from file in yaml format
*/
class SimulationLoader
{
public:
	SimulationLoader();
	bool Load(const std::string& filepath);

	inline std::vector<obstacle> GetObstacles() const { return m_Obstacles; };
	inline flow_area GetInArea() const { return m_InArea; };
	inline flow_area GetOutArea() const { return m_OutArea; };
	inline const physicsSpecification& GetSpecification() const { return m_PhysicsSpec; };
	inline uint32_t GetParticleLimit() const { return m_ParticleLimit; };

private:
	void CreateDefoultConfiguration();

private:
	std::vector<obstacle> m_Obstacles;
	flow_area m_InArea;
	flow_area m_OutArea;
	physicsSpecification m_PhysicsSpec;
	uint32_t m_ParticleLimit;
};
