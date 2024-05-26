#pragma once
#include <vector>

#include "Cuda/Kernels.h"

#include "Core/Base.h"

#include "Physics/PhysicsSpecification.h"
#include "Physics/Particles.h"

#include "iostream"

class CollisionHandler
{
public:
	CollisionHandler(PhysicsSpecification& spec, std::vector<obstacle>& obstacles) : p_spec(spec) {
		CUDA_CALL(cudaMalloc(&c_obstacles_addr, obstacles.size() * sizeof(obstacle)));
		CUDA_CALL(cudaMemcpy(c_obstacles_addr, obstacles.data(), obstacles.size() * sizeof(obstacle), cudaMemcpyHostToDevice));
		obstacles_size = obstacles.size();
	}

	void Resolve(Ref<Particles> particles, float deltaTime) const;

private:
	obstacle* c_obstacles_addr;
	size_t obstacles_size = 0;
	PhysicsSpecification& p_spec;
};

