#include "Particle.h"


void Particle::Update(float deltaTime, float environmentScale) {
	ApplyForce(deltaTime);
	ResetTemporaryProperties();
	UpdatePosition(deltaTime, environmentScale);
}

void Particle::AddForce(const glm::vec2& force) {
	this->force += force;
}

void Particle::AddPartialDensity(const float density) {
	this->density += density;
}

glm::vec2 Particle::GetPosition()const
{
	return this->position;
};

glm::vec2 Particle::GetVelocity()const
{
	return this->velocity;
}

float Particle::GetDensity()const {
	return this->density;
}

float Particle::GetPressure()const {
	return this->pressure;
}

void Particle::SetPosition(const glm::vec2& position) {
	this->position = position;
}

void Particle::SetVelocity(const glm::vec2& velocity) {
	this->velocity = velocity;
}

void Particle::SetPressure(const float pressure) {
	this->pressure = pressure;
}

void Particle::ApplyForce(float deltaTime) {
	this->velocity += this->force * deltaTime / this->density;
}

void Particle::ResetTemporaryProperties() {
	this->force = glm::vec2(0, 0);
	this->density = 0.0f;
	this->pressure = 0.0f;
}

void Particle::UpdatePosition(float deltaTime, float environmentScale) {
	this->position += this->velocity * deltaTime * environmentScale;
}

float Particle::calculateDistance(const Particle& otherParticle)const {
	return sqrt(pow(this->position.x - otherParticle.GetPosition().x, 2) +
		pow(this->position.y - otherParticle.GetPosition().y, 2));
}
