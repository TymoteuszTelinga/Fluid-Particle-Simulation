#pragma once

#include <glm/glm.hpp>

#include<stdio.h>
#include <vector>

#include "Core/Base.h"

#include "Physics/Properties/Density.h"

#include "Physics/Forces/Gravity.h"
#include "Physics/Forces/Pressure.h"

struct PhysicsSpecification {
	uint32_t Width;
	uint32_t Height;
};

class Physics
{
public:
	static constexpr float PixelToMeters = 0.01f; // 1 px = 1cm = 0.01m
	static constexpr float MeterToPixels = 100.0f; // 1m = 100 px

public:
	Physics(PhysicsSpecification& spec): m_specification(spec) {
		l_Gravity = CreateScope<Gravity>();
		l_Density = CreateScope<Density>();
		l_Pressure = CreateScope<Pressure>();
	}

	void Apply(std::vector<Particle>& particles, const float deltaTime) const;

private:
	void BounceFromBorder(Particle& particle) const;

private:
	PhysicsSpecification m_specification;
	Scope<Gravity> l_Gravity;
	Scope<Pressure> l_Pressure;
	Scope<Density> l_Density;
	
};

