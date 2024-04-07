#pragma once
#include <glm/glm.hpp>



class Particle
{
public:
	static constexpr float PARTICLE_RADIUS = 2.5f;

	Particle(float xPos, float  yPos)
		: position(xPos, yPos), velocity(0, 0), force(0, 0) {};

	void Update(float deltaTime) {
		ApplyForce(deltaTime);
		UpdatePosition(deltaTime);
	}

	void AddForce(const glm::vec2& force) {
		this->force += force;
	}


	glm::vec2 GetPosition()const
	{
		return this->position;
	};

	glm::vec2 GetVelocity()const
	{
		return this->velocity;
	}

	void SetPosition(const glm::vec2& position) {
		this->position = position;
	}

	void SetVelocity(const glm::vec2& velocity) {
		this->velocity = velocity;
	}

private:
	void ApplyForce(float deltaTime) {
		this->velocity += this->force * deltaTime;
		this->force = glm::vec2(0, 0);
	}

	void UpdatePosition(float deltaTime) {
		this->position += this->velocity * deltaTime;
	}


private:
	glm::vec2 position;
	glm::vec2 velocity;
	glm::vec2 force; 
};

