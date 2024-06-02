#include "SimulationLoader.h"
#include <fstream>
#include <yaml-cpp/yaml.h>

static void serializeObstacle(YAML::Emitter& out, const obstacle& o)
{
	out << YAML::BeginMap;
	out << YAML::Key << "Min" << YAML::Value << YAML::Flow << YAML::BeginSeq << o.x_pos << o.y_pos << YAML::EndSeq;
	out << YAML::Key << "Max" << YAML::Value << YAML::Flow << YAML::BeginSeq << o.x_pos+o.width << o.y_pos+o.height << YAML::EndSeq;
	out << YAML::EndMap;
};

static void LoadSpecification(const YAML::Node& node, physicsSpecification& spec)
{
	spec.Width = node["Width"].as<float>();
	spec.Height = node["Height"].as<float>();
	spec.GravityAcceleration = node["Gravity"].as<float>();
	spec.CollisionDamping = node["CollisionDamping"].as<float>();
	spec.KernelRange = node["KernelRange"].as<float>();
	spec.RestDensity = node["RestDensity"].as<float>();
	spec.GasConstant = node["GasConstant"].as<float>();
	spec.NearPressureCoef = node["NearPressureCoef"].as<float>();
	spec.ViscosityStrength = node["Viscosity"].as<float>();
}

SimulationLoader::SimulationLoader()
{
	CreateDefoultConfiguration();
}

bool SimulationLoader::Load(const std::string& filepath)
{
	m_Obstacles.clear();

	YAML::Node data = YAML::LoadFile(filepath);
	if (!data["Obstacles"])
	{
		return false;
	}

	auto objects = data["Obstacles"];
	for (auto object : objects)
	{
		obstacle o;

		const YAML::Node& min = object["Min"];
		o.x_pos = min[0].as<float>();
		o.y_pos = min[1].as<float>();
		
		const YAML::Node& max = object["Max"];
		o.width = max[0].as<float>() - o.x_pos;
		o.height = max[1].as<float>() - o.y_pos;

		m_Obstacles.push_back(o);
	}

	auto in = data["InArea"];
	if (in)
	{
		const YAML::Node& min = in["Min"];
		m_InArea.x_pos = min[0].as<float>();
		m_InArea.y_pos = min[1].as<float>();

		const YAML::Node& max = in["Max"];
		m_InArea.width = max[0].as<float>() - m_InArea.x_pos;
		m_InArea.height = max[1].as<float>() - m_InArea.y_pos;
	}
	else
	{
		return false;
	}

	auto out = data["OutArea"];
	if (out)
	{
		const YAML::Node& min = out["Min"];
		m_OutArea.x_pos = min[0].as<float>();
		m_OutArea.y_pos = min[1].as<float>();

		const YAML::Node& max = out["Max"];
		m_OutArea.width = max[0].as<float>() - m_OutArea.x_pos;
		m_OutArea.height = max[1].as<float>() - m_OutArea.y_pos;
	}
	else
	{
		return false;
	}

	auto physics = data["Physics"];
	if (physics)
	{
		LoadSpecification(physics, m_PhysicsSpec);
		m_ParticleLimit = physics["ParticleLimit"].as<uint32_t>();
	}
	else
	{
		CreateDefoultConfiguration();
	}

	return true;

	/*
	YAML::Emitter out;

	out << YAML::BeginMap;
	out << YAML::Key << "Obstacles" << YAML::Value << YAML::BeginSeq;
	for (auto& obs : m_Obstacles)
	{
		serializeObstacle(out, obs);
	}

	out << YAML::EndSeq;
	out << YAML::EndMap;

	std::ofstream fout(filepath);
	fout << out.c_str();
	*/
}

void SimulationLoader::CreateDefoultConfiguration()
{
	m_PhysicsSpec.Width = 19.f;
	m_PhysicsSpec.Height = 11.f;
	m_PhysicsSpec.GravityAcceleration = 9.81f;
	m_PhysicsSpec.CollisionDamping = 0.2;
	m_PhysicsSpec.KernelRange = 0.35;
	m_PhysicsSpec.RestDensity = 55;
	m_PhysicsSpec.GasConstant = 500;
	m_PhysicsSpec.NearPressureCoef = 18;
	m_PhysicsSpec.ViscosityStrength = 0.06;

	m_ParticleLimit = 2000;
}
