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
