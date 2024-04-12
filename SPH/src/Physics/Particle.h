#pragma once
#include <glm/glm.hpp>

class Particle
{
public:
	Particle(float xPos, float yPos)
		: position(xPos, yPos), velocity(0.0f, 0.0f), force(0.0f, 0.0f), density(0.0f), pressure(0.0f) {};

	void Update(float deltaTime);
	void AddForce(const glm::vec2& force);
	void AddPartialDensity(const float density);
	void SetPressure(const float pressure);

	glm::vec2 GetPosition()const;
	glm::vec2 GetVelocity()const;
	float GetDensity()const;
	float GetPressure()const;

	void SetPosition(const glm::vec2& position);
	void SetVelocity(const glm::vec2& velocity);

	float calculateDistance(const Particle& otherParticle)const;

private:
	void ResetTemporaryProperties();
	void ApplyForce(float deltaTime);
	void UpdatePosition(float deltaTime);

private:
	glm::vec2 position;
	glm::vec2 velocity;
	glm::vec2 force; 

	float density;
	float pressure;
};

