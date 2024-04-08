#pragma once

#include <glm/glm.hpp>

#include<stdio.h>
#include <vector>

#include "Physics/Particle.h"

class Gravity
{

public:
	void Apply(std::vector<Particle>& particles, float deltaTime) const;

private:
	static constexpr float FORCE = 10.0f; // m/s^2
};

