#include "NeighbourSearch.h"

#include <algorithm>

#include "Physics/Cuda/Kernels.h"
#include "iostream"

void NeighbourSearch::UpdateSpatialLookup(Ref<Particles> particles) {
	PrepareLookup(particles->getSize());
	FillSpatialLookup(particles);
	SortLookUp();
	FillStartIndices();

	for (int i = 0; i < particles->getSize(); i++) {
		this->spatialLookupIndex->operator[](i) = this->spatialLookup->operator[](i).particleIndex;
		this->spatialLookupKey->operator[](i) = this->spatialLookup->operator[](i).cellKey;
	}

	CUDA_CALL(cudaFree(particles->c_indices_addr));
	CUDA_CALL(cudaMalloc(&particles->c_indices_addr, this->startIndices->size() * sizeof(int)));
	CUDA_CALL(cudaMemcpy(particles->c_lookup_indexes_addr, this->spatialLookupIndex->data(), particles->getSize()*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(particles->c_lookup_keys_addr, this->spatialLookupKey->data(), particles->getSize()*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(particles->c_indices_addr, this->startIndices->data(), this->startIndices->size()*sizeof(int), cudaMemcpyHostToDevice));
	particles->c_indices_size = this->startIndices->size();
}

void NeighbourSearch::PrepareLookup(size_t lookupSize) {
	this->spatialLookup = CreateScope<std::vector<LookupEntry>>(lookupSize, LookupEntry{ 0,0 });
	this->spatialLookupIndex = CreateScope<std::vector<int>>(lookupSize, 0);
	this->spatialLookupKey = CreateScope<std::vector<int>>(lookupSize, 0);

	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	int cellCols = (int)(p_spec.Height / p_spec.KernelRange) + 1;
	this->startIndices = CreateScope<std::vector<int>>(cellRows * cellCols, 0);
}

Ref<std::vector<size_t>> NeighbourSearch::GetParticleNeighbours(Ref<Particles> particles, size_t particleIndex)const {
	Ref<std::vector<size_t>> neighbours = CreateRef<std::vector<size_t>>();
	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	size_t cellKey = PositionToCellKey(particles->getPredictedPosition(particleIndex));

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


void NeighbourSearch::AddNeighbours(Ref<std::vector<size_t>> neighbours, Ref<Particles> particles, size_t particleIndex, int cellKey) const{
	if (cellKey < 0 || cellKey >= startIndices->size()) {
		return;
	}

	size_t startIndice = startIndices->operator[](cellKey);

	float sqrRange = p_spec.KernelRange * p_spec.KernelRange;
	for (size_t i = startIndice; i < spatialLookup->size(); i++) {
		if (spatialLookup->operator[](i).cellKey != cellKey) {
			break;
		}

		size_t selectedParticleIndex = spatialLookup->operator[](i).particleIndex;

		float distance = particles->calculatePredictedDistance(selectedParticleIndex, particleIndex);

		if (distance*distance <= sqrRange) {
			neighbours->push_back(selectedParticleIndex);
		}
	}
}

void NeighbourSearch::FillSpatialLookup(Ref<Particles> particles) {
	for (size_t i = 0; i < particles->getSize(); i++) {
		size_t cellKey = PositionToCellKey(particles->getPredictedPosition(i));
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

		if (cellKey != prevKey) {
			startIndices->operator[](cellKey) = i;
		}
		prevKey = cellKey;
	}
}

size_t NeighbourSearch::PositionToCellKey(glm::vec2 position) const {
	int cellRows = (int)(p_spec.Width / p_spec.KernelRange) + 1;
	int cellCols = (int)(p_spec.Height / p_spec.KernelRange) + 1;

	int cellX = (int)(position.x / p_spec.KernelRange);
	int cellY = (int)(position.y / p_spec.KernelRange);

	cellX = std::max(0,std::min(cellRows - 1, cellX));
	cellY = std::max(0, std::min(cellCols - 1, cellY));

	return cellX + cellY * cellRows;
}