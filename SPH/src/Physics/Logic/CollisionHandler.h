#pragma once
#include <vector>

#include "Core/Base.h"

#include "Physics/Cuda/Kernels.h"
#include "Physics/physicsSpecification.h"
#include "Physics/Entities/Particles.h"

#include "iostream"

class CollisionHandler
{
public:
	CollisionHandler(physicsSpecification& spec, std::vector<obstacle>& obstacles) : p_spec(spec) {
		CUDA_CALL(cudaMalloc(&c_obstacles_addr, obstacles.size() * sizeof(obstacle)));
		CUDA_CALL(cudaMemcpy(c_obstacles_addr, obstacles.data(), obstacles.size() * sizeof(obstacle), cudaMemcpyHostToDevice));
		obstacles_size = obstacles.size();
	}

	void Resolve(Ref<Particles> particles, float deltaTime) const;

private:
	obstacle* c_obstacles_addr;
	size_t obstacles_size = 0;
	physicsSpecification& p_spec;
};

