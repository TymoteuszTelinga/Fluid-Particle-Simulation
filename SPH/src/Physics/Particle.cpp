#include "Particle.h"


void Particle::Update(float deltaTime) {
	ApplyForce(deltaTime);
	UpdatePosition(deltaTime);
}

void Particle::AddForce(const glm::vec2& force) {
	this->force += force;
}

glm::vec2 Particle::GetPosition()const
{
	return this->position;
};

glm::vec2 Particle::GetVelocity()const
{
	return this->velocity;
}

void Particle::SetPosition(const glm::vec2& position) {
	this->position = position;
}

void Particle::SetVelocity(const glm::vec2& velocity) {
	this->velocity = velocity;
}

void Particle::ApplyForce(float deltaTime) {
	this->velocity += this->force * deltaTime;
	this->force = glm::vec2(0, 0);
}

void Particle::UpdatePosition(float deltaTime) {
	this->position += this->velocity * deltaTime;
}
