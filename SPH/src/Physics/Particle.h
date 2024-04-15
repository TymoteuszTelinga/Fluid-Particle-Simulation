#pragma once
#include <glm/glm.hpp>

class Particle
{
public:
	Particle(float xPos, float yPos)
		: position(xPos, yPos), velocity(0.0f, 0.0f), force(0.0f, 0.0f), density(0.0f), predicted_position(xPos, yPos){};

	void Update(float deltaTime);
	void AddForce(const glm::vec2& force);
	void AddPartialDensity(const float density);
	void PredictionUpdate(float deltaTime);

	glm::vec2 GetPosition()const;
	glm::vec2 GetPredictedPosition()const;
	glm::vec2 GetVelocity()const;
	float GetDensity()const;

	void SetPosition(const glm::vec2& position);
	void SetVelocity(const glm::vec2& velocity);

	float calculateDistance(const Particle& otherParticle)const;

private:
	void ResetTemporaryProperties();
	void ApplyForce(float deltaTime);
	void UpdatePosition(float deltaTime);
	void PredictPosition(float deltaTime);

private:
	glm::vec2 position;
	glm::vec2 predicted_position;
	glm::vec2 velocity;
	glm::vec2 force; 

	float density;
};

