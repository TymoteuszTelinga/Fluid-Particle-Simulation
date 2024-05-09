#include "Particle.h"


void Particle::Update(float deltaTime) {
	ApplyForce(deltaTime);
	ResetTemporaryProperties();
	UpdatePosition(deltaTime);
}

void Particle::PredictionUpdate(float deltaTime) {
	ApplyForce(deltaTime);
	ResetTemporaryProperties();
	PredictPosition(deltaTime);
}

void Particle::PredictPosition(float deltaTime) {
	predicted_position = position + velocity * deltaTime;
}

void Particle::AddForce(const glm::vec2& force) {
	this->force += force;
}

void Particle::AddPartialDensity(const float density) {
	this->density += density;
}

void Particle::AddPartialNearDensity(const float nearDensity){
	this->nearDensity += nearDensity;
}

glm::vec2 Particle::GetPosition()const
{
	return this->position;
}; 

glm::vec2 Particle::GetPredictedPosition()const
{
	return this->predicted_position;
};

glm::vec2 Particle::GetVelocity()const
{
	return this->velocity;
}

float Particle::GetDensity()const {
	return this->density;
}

float Particle::GetNearDensity() const
{
	return this->nearDensity;
}

void Particle::SetPosition(const glm::vec2& position) {
	this->position = position;
}

void Particle::SetVelocity(const glm::vec2& velocity) {
	this->velocity = velocity;
}

void Particle::ApplyForce(float deltaTime) {
	this->velocity += this->force * deltaTime;
}

void Particle::ResetTemporaryProperties() {
	this->force = glm::vec2(0, 0);
	this->density = 0.0f;
	this->nearDensity = 0.0f;
}

void Particle::UpdatePosition(float deltaTime) {
	this->position += this->velocity * deltaTime;
}

float Particle::calculatePredictedDistance(const Particle& otherParticle)const {
	return sqrt(pow(this->predicted_position.x - otherParticle.GetPredictedPosition().x, 2) +
		pow(this->predicted_position.y - otherParticle.GetPredictedPosition().y, 2));
}
