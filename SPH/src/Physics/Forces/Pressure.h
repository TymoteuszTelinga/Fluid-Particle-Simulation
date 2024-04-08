#pragma once
#include <vector>

#include <glm/glm.hpp>

#include "Physics/Particle.h"
#include "Physics/Kernel.h"

class Pressure
{
public:
	void Apply(std::vector<Particle>& particles, float deltaTime);

private:
	void CalculatePressures(std::vector<Particle>& particles);
	float CalculatePressure(const Particle& particle);

private:
	static constexpr float KERNEL_RADIUS = 20.0f; // 20cm
	static constexpr KernelFunc KERNEL_DERIV = Kernel::Spiky3Deriv;
	static constexpr float GAS_CONSTANT = 10000.0f;
};

