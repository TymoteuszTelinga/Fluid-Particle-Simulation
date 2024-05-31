#pragma once
#include "Core/ComonTypes.h"
#include "Physics/PhysicsSpecification.h"
#include <string>
#include <vector>

class SimulationLoader
{
public:
	SimulationLoader() {};
	bool Load(const std::string& filepath);

	inline std::vector<obstacle> GetObstacles() const { return m_Obstacles; };
	inline flow_area GetInArea() const { return m_InArea; };
	inline flow_area GetOutArea() const { return m_OutArea; };
	inline PhysicsSpecification GetSpecification() const { return m_PhysicsSpec; };

private:
	std::vector<obstacle> m_Obstacles;
	flow_area m_InArea;
	flow_area m_OutArea;
	PhysicsSpecification m_PhysicsSpec;
};
