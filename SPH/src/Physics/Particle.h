#pragma once
#include <glm/glm.hpp>



class Particle
{
public:
	static constexpr float PARTICLE_RADIUS = 2.5f;

	Particle(float xPos, float  yPos)
		: position(xPos, yPos), velocity(0, 0), force(0, 0) {};

	void Update(float deltaTime);
	void AddForce(const glm::vec2& force);

	glm::vec2 GetPosition()const;
	glm::vec2 GetVelocity()const;

	void SetPosition(const glm::vec2& position);
	void SetVelocity(const glm::vec2& velocity);

private:
	void ApplyForce(float deltaTime);
	void UpdatePosition(float deltaTime);


private:
	glm::vec2 position;
	glm::vec2 velocity;
	glm::vec2 force; 
};

