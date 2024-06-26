#pragma once

#include <vector>


#include "Core/Base.h"

#include "Physics/physicsSpecification.h"
#include "Physics/Entities/Particles.h"

struct LookupEntry {
	size_t particleIndex = 0;
	size_t cellKey = 0;
};

class NeighbourSearch
{
public:
	NeighbourSearch(physicsSpecification& spec) : p_spec(spec) {};

	void UpdateSpatialLookup(Ref<Particles> particles);
	Ref<std::vector<size_t>> GetParticleNeighbours(Ref<Particles> particles, size_t particleIndex)const;

private:
	void PrepareLookup(size_t lookupSize);
	void FillSpatialLookup(Ref<Particles> particles);
	void SortLookUp();
	void FillStartIndices();
	void AddNeighbours(Ref<std::vector<size_t>> neighbours, Ref<Particles> particles, size_t particleIndex, int cellKey)const;

	size_t PositionToCellKey(glm::vec2 position) const;
	

private:
	physicsSpecification& p_spec;
	Scope<std::vector<LookupEntry>> spatialLookup;
	Scope<std::vector<int>> spatialLookupIndex;
	Scope<std::vector<int>> spatialLookupKey;
	Scope<std::vector<int>> startIndices;

};

