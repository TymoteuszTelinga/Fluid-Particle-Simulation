#include "NeighbourSearch.h"


void NeighbourSearch::UpdateSpatialLookup(std::vector<Particle>& particles) {
	PrepareLookup(particles.size());
	FillSpatialLookup(particles);
	SortLookUp();
	FillStartIndices();
}

Ref<std::vector<size_t>> NeighbourSearch::GetParticleNeighbours(std::vector<Particle>& particles, size_t particleIndex) {
	Ref<std::vector<size_t>> neighbours = CreateRef<std::vector<size_t>>();
	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	size_t cellKey = PositionToCellKey(particles[particleIndex].GetPredictedPosition());

	AddNeighbours(neighbours, particles, particleIndex, cellKey - cellRows - 1);
	AddNeighbours(neighbours, particles, particleIndex, cellKey - cellRows);
	AddNeighbours(neighbours, particles, particleIndex, cellKey - cellRows + 1);
	AddNeighbours(neighbours, particles, particleIndex, cellKey-1);
	AddNeighbours(neighbours, particles, particleIndex, cellKey);
	AddNeighbours(neighbours, particles, particleIndex, cellKey + 1);
	AddNeighbours(neighbours, particles, particleIndex, cellKey + cellRows - 1);
	AddNeighbours(neighbours, particles, particleIndex, cellKey + cellRows);
	AddNeighbours(neighbours, particles, particleIndex, cellKey + cellRows + 1);

	return neighbours;
}


void NeighbourSearch::AddNeighbours(Ref<std::vector<size_t>> neighbours, std::vector<Particle>& particles, size_t particleIndex, size_t cellKey) {
	if (cellKey < 0 || cellKey >= startIndices->size()) {
		return;
	}

	size_t startIndice = startIndices->operator[](cellKey);
	float sqrtRange = p_spec.KernelRange * p_spec.KernelRange;
	for (size_t i = startIndice; i < spatialLookup->size(); i++) {
		if (spatialLookup->operator[](i).cellKey != cellKey) {
			break;
		}

		int selectedParticleIndex = spatialLookup->operator[](i).particleIndex;
		float sqrtDistance = particles[selectedParticleIndex].calculateDistance(particles[particleIndex]);

		if (sqrtDistance < sqrtRange) {
			neighbours->push_back(selectedParticleIndex);
		}
	}
}

Ref<std::vector<LookupEntry>> NeighbourSearch::GetSpatialLookup()const {
	return this->spatialLookup;
}

Ref<std::vector<int>> NeighbourSearch::GetStartIndices() const {
	return this->startIndices;
}

void NeighbourSearch::PrepareLookup(size_t lookupSize) {
	this->spatialLookup = CreateRef<std::vector<LookupEntry>>(lookupSize, LookupEntry{ 0,0 });

	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	int cellCols = (int)(p_spec.Height / p_spec.KernelRange) + 1;
	this->startIndices = CreateRef<std::vector<int>>(cellRows * cellCols, 0);
}

void NeighbourSearch::FillSpatialLookup(std::vector<Particle>& particles) {
	for (size_t i = 0; i < particles.size(); i++) {
		Particle& p = particles[i];
		size_t cellKey = PositionToCellKey(p.GetPredictedPosition());
		this->spatialLookup->operator[](i) = LookupEntry{ i, cellKey };
	}
}

void NeighbourSearch::SortLookUp() {
	std::sort(this->spatialLookup->begin(), this->spatialLookup->end(), [](const LookupEntry& a, const LookupEntry& b) {
		return a.cellKey < b.cellKey;
		});
}

void NeighbourSearch::FillStartIndices() {
	size_t prevKey = UINT64_MAX;
	for (int i = 0; i < spatialLookup->size(); i++) {
		size_t cellKey = spatialLookup->operator[](i).cellKey;
		if (cellKey >= startIndices->size()) {
			cellKey = startIndices->size() - 1;
		}
		if (cellKey != prevKey) {
			startIndices->operator[](cellKey) = i;
		}
		prevKey = cellKey;
	}
}

size_t NeighbourSearch::PositionToCellKey(glm::vec2 position) const {
	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	int cellX = (int)(position.x / p_spec.KernelRange);
	int cellY = (int)(position.y / p_spec.KernelRange);

	return cellX + cellY * cellRows;
}