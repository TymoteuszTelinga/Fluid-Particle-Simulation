#pragma once

#include <vector>
#include <algorithm>

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particle.h"

struct LookupEntry {
	size_t particleIndex = 0;
	size_t cellKey = 0;
};

class NeighbourSearch
{
public:
	NeighbourSearch(PhysicsSpecification& spec) : p_spec(spec) {};

	void UpdateSpatialLookup(std::vector<Particle>& particles);
	Ref<std::vector<size_t>> GetParticleNeighbours(std::vector<Particle>& particles, size_t particleIndex);

	Ref<std::vector<LookupEntry>> GetSpatialLookup()const;
	Ref<std::vector<int>> GetStartIndices() const;

private:
	void PrepareLookup(size_t lookupSize);
	void FillSpatialLookup(std::vector<Particle>& particles);
	void SortLookUp();
	void FillStartIndices();
	void AddNeighbours(Ref<std::vector<size_t>> neighbours, std::vector<Particle>& particles, size_t particleIndex, size_t cellKey);

	size_t PositionToCellKey(glm::vec2 position) const;
	

private:
	PhysicsSpecification& p_spec;
	Ref<std::vector<LookupEntry>> spatialLookup;
	Ref<std::vector<int>> startIndices;
};

