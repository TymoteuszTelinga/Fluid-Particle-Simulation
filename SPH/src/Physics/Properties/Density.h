#pragma once
#include <vector>

#include <glm/glm.hpp>

#include "Physics/Particle.h"
#include "Physics/Kernel.h"

class Density
{

public:
	void Calculate(std::vector<Particle>& particles);
	static constexpr float REST_DENSITY = 0.0f;

private:
	static constexpr float KERNEL_RADIUS = 20.0f; // 20cm
	static constexpr KernelFunc KERNEL = Kernel::Poly6;
};

